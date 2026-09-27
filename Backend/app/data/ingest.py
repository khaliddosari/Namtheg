"""Convert any supported upload into a typed Parquet DataFrame.

This runs exactly once per upload. Every later stage (profiling, the analyst
agent's sandbox, training) loads the Parquet copy instead of re-parsing the raw
file, so dtypes are decided once and a stage load costs milliseconds instead of
a full CSV parse.

Normalisation is deliberately conservative: values are never re-interpreted
(no guessing that "1,234" is a number or that "01/02/2024" is a date). Anything
that does change the data is listed in the report's `actions` so downstream
consumers, and the user, can see exactly what happened. The one convention
applied silently is pandas' standard missing-value tokens ("", "NA", "N/A",
"null", "NaN", "#N/A", ...), which are read as missing.

Script usage:
    python -m app.data.ingest input.xlsx output.parquet
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

SUPPORTED_EXTENSIONS = (".csv", ".tsv", ".txt", ".xlsx", ".xls", ".parquet", ".json", ".jsonl")

_SNIFF_BYTES = 64 * 1024
_CSV_ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")  # latin-1 never fails, so it's the last resort


class IngestError(ValueError):
    """The file can't be turned into a table. The message is safe to show the user."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sniff_delimiter(path: Path, ext: str) -> str:
    if ext == ".tsv":
        return "\t"
    sample = path.read_bytes()[:_SNIFF_BYTES].decode("utf-8", errors="replace")
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def _read_delimited(path: Path, ext: str, report: dict) -> pd.DataFrame:
    delimiter = _sniff_delimiter(path, ext)
    report["delimiter"] = delimiter
    for encoding in _CSV_ENCODINGS:
        try:
            df = pd.read_csv(path, sep=delimiter, encoding=encoding, low_memory=False)
        except UnicodeDecodeError:
            continue
        report["encoding"] = encoding
        if encoding != "utf-8-sig":
            report["warnings"].append(f"File is not valid UTF-8; decoded as {encoding}.")
        return df
    raise IngestError("Could not decode the file's text encoding.")


def _read_excel(path: Path, report: dict) -> pd.DataFrame:
    sheets = pd.read_excel(path, sheet_name=None, engine="calamine")
    non_empty = [(name, df) for name, df in sheets.items() if not df.dropna(how="all").empty]
    if not non_empty:
        raise IngestError("The workbook has no non-empty sheets.")
    name, df = non_empty[0]
    report["sheet"] = str(name)
    if len(non_empty) > 1:
        others = ", ".join(repr(str(n)) for n, _ in non_empty[1:])
        report["warnings"].append(
            f"Workbook has {len(non_empty)} non-empty sheets; used the first ({name!r}) and ignored {others}."
        )
    return df


def _read_json(path: Path, ext: str) -> pd.DataFrame:
    if ext == ".jsonl":
        return pd.read_json(path, lines=True)
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, list):
        # Flattens nested objects into dotted columns: {"a": {"b": 1}} -> "a.b".
        return pd.json_normalize(payload)
    if isinstance(payload, dict):
        list_values = [v for v in payload.values() if isinstance(v, list)]
        if len(list_values) == 1 and all(isinstance(r, dict) for r in list_values[0]):
            return pd.json_normalize(list_values[0])
        return pd.DataFrame(payload)
    raise IngestError("JSON must be a list of records or an object of columns.")


def read_any(path: Path, filename: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Parse `path` into a DataFrame. `filename` (the user's original name)
    decides the format when the stored file has a generic name."""
    ext = Path(filename or path.name).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise IngestError(
            f"Unsupported file type '{ext or '(none)'}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}."
        )
    if path.stat().st_size == 0:
        raise IngestError("The file is empty.")

    report: dict = {
        "source_filename": filename or path.name,
        "source_format": ext.lstrip("."),
        "source_bytes": path.stat().st_size,
        "source_sha256": _sha256(path),
        "encoding": None,
        "delimiter": None,
        "sheet": None,
        "actions": [],
        "warnings": [],
    }
    try:
        if ext in (".csv", ".tsv", ".txt"):
            df = _read_delimited(path, ext, report)
        elif ext in (".xlsx", ".xls"):
            df = _read_excel(path, report)
        elif ext == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = _read_json(path, ext)
    except IngestError:
        raise
    except Exception as e:
        raise IngestError(f"Could not parse the file as {ext.lstrip('.').upper()}: {e}") from e
    return df, report


def _dedupe_column_names(df: pd.DataFrame, actions: list[str]) -> pd.DataFrame:
    seen: dict[str, int] = {}
    names: list[str] = []
    for i, raw in enumerate(df.columns):
        name = str(raw).strip()
        if not name:
            name = f"column_{i + 1}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        seen.setdefault(name, 1)
        if name != str(raw):
            actions.append(f"Renamed column {str(raw)!r} to {name!r}.")
        names.append(name)
    df.columns = names
    return df


def _coerce_object_column(s: pd.Series) -> tuple[pd.Series, str | None]:
    """Make an object column storable in Parquet without reinterpreting text.

    Only columns holding non-string Python objects are touched: numbers stay
    numbers, dates become datetimes, and anything genuinely mixed (e.g. an
    Excel column with both 12 and "N/A") is stored as text.
    """
    kind = pd.api.types.infer_dtype(s, skipna=True)
    if kind in ("string", "empty", "boolean"):
        return s, None
    try:
        if kind in ("integer", "floating", "mixed-integer-float", "decimal"):
            return pd.to_numeric(s), f"stored as numeric (held {kind} objects)"
        if kind in ("datetime", "datetime64", "date"):
            return pd.to_datetime(s), f"stored as datetime (held {kind} objects)"
    except (TypeError, ValueError):
        pass  # e.g. mixed timezone-aware and naive datetimes: keep as text below
    return s.map(lambda v: v if pd.isna(v) else str(v)), f"stored as text (held mixed types: {kind})"


def normalize(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    actions = report["actions"]
    df = _dedupe_column_names(df, actions)

    empty_cols = [c for c in df.columns if df[c].isna().all()]
    if empty_cols:
        df = df.drop(columns=empty_cols)
        actions.append(f"Dropped {len(empty_cols)} completely empty column(s): {empty_cols}.")

    n_before = len(df)
    df = df.dropna(how="all").reset_index(drop=True)
    if len(df) < n_before:
        actions.append(f"Dropped {n_before - len(df)} completely empty row(s).")

    for c in df.columns:
        if isinstance(df[c].dtype, pd.CategoricalDtype):
            df[c] = df[c].astype(object)
            actions.append(f"Column {c!r}: categorical dtype stored as plain values.")
        elif df[c].dtype == object:
            df[c], note = _coerce_object_column(df[c])
            if note:
                actions.append(f"Column {c!r}: {note}.")

    if df.empty or df.shape[1] == 0:
        raise IngestError("The file contains no data rows.")
    return df


def ingest_file(src: Path, dest: Path, filename: str | None = None) -> dict:
    """Parse `src`, normalise it, and write it to `dest` as Parquet.

    Returns the ingest report (source metadata, final shape and dtypes, and
    every action taken). Raises IngestError with a user-facing message.
    """
    df, report = read_any(src, filename)
    df = normalize(df, report)
    tmp = dest.with_suffix(".tmp")
    df.to_parquet(tmp, index=False, compression="zstd")
    tmp.replace(dest)
    report.update({
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "parquet_bytes": dest.stat().st_size,
    })
    return report


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python -m app.data.ingest <input file> <output.parquet>", file=sys.stderr)
        return 2
    try:
        report = ingest_file(Path(argv[0]), Path(argv[1]))
    except IngestError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
