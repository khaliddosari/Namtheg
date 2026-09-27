import json

import pandas as pd
import pytest

from app.data.ingest import IngestError, ingest_file, main


def _ingest(tmp_path, name: str, write) -> tuple[pd.DataFrame, dict]:
    src = tmp_path / name
    write(src)
    dest = tmp_path / "out.parquet"
    report = ingest_file(src, dest)
    return pd.read_parquet(dest), report


def test_csv_types_are_preserved(tmp_path):
    df, report = _ingest(tmp_path, "a.csv", lambda p: p.write_text("id,price,city\n1,9.5,Riyadh\n2,,Jeddah\n"))
    assert list(df.columns) == ["id", "price", "city"]
    assert df["id"].dtype.kind == "i" and df["price"].dtype.kind == "f"
    assert report["n_rows"] == 2 and report["delimiter"] == "," and report["encoding"] == "utf-8-sig"
    assert report["actions"] == []


@pytest.mark.parametrize("sep", [";", "\t", "|"])
def test_delimiter_is_sniffed(tmp_path, sep):
    df, report = _ingest(tmp_path, "a.csv", lambda p: p.write_text(f"a{sep}b\n1{sep}x\n2{sep}y\n3{sep}z\n"))
    assert list(df.columns) == ["a", "b"] and report["delimiter"] == sep


def test_non_utf8_is_decoded_and_warned(tmp_path):
    df, report = _ingest(tmp_path, "a.csv", lambda p: p.write_bytes("name,v\nJos\xe9,1\n".encode("cp1252")))
    assert df.loc[0, "name"] == "José"
    assert report["encoding"] == "cp1252" and report["warnings"]


def test_arabic_utf8_with_bom(tmp_path):
    df, _ = _ingest(tmp_path, "a.csv", lambda p: p.write_bytes("﻿المدينة,القيمة\nالرياض,5\n".encode("utf-8")))
    assert list(df.columns) == ["المدينة", "القيمة"]


def test_numbers_as_text_are_not_reinterpreted(tmp_path):
    df, _ = _ingest(tmp_path, "a.csv", lambda p: p.write_text('amount,d\n"1,234",01/02/2024\n"5,000",03/04/2024\n'))
    assert df["amount"].tolist() == ["1,234", "5,000"]
    assert df["d"].tolist() == ["01/02/2024", "03/04/2024"]


def test_empty_rows_columns_and_messy_headers(tmp_path):
    df, report = _ingest(tmp_path, "a.csv", lambda p: p.write_text(" a ,b,,\n1,2,,\n,,,\n3,4,,\n"))
    assert list(df.columns) == ["a", "b"]
    assert len(df) == 2
    assert any("Renamed column ' a '" in a for a in report["actions"])
    assert any("empty column" in a for a in report["actions"])
    assert any("empty row" in a for a in report["actions"])


def test_excel_mixed_type_column_is_stored_as_text(tmp_path):
    def write(p):
        pd.DataFrame({
            "x": [1, "unknown", 3],
            "y": [1.5, 2.5, 3.5],
            "n": [1, "N/A", 3],  # standard missing-value token: numeric with a gap
        }).to_excel(p, index=False, sheet_name="data")
        with pd.ExcelWriter(p, mode="a", engine="openpyxl") as w:
            pd.DataFrame({"z": [1]}).to_excel(w, index=False, sheet_name="notes")

    df, report = _ingest(tmp_path, "a.xlsx", write)
    assert df["x"].tolist() == ["1", "unknown", "3"]
    assert df["y"].dtype.kind == "f"
    assert df["n"].isna().tolist() == [False, True, False]
    assert report["sheet"] == "data"
    assert any("2 non-empty sheets" in w for w in report["warnings"])


def test_json_records_are_flattened(tmp_path):
    rows = [{"a": 1, "meta": {"b": "x"}}, {"a": 2, "meta": {"b": "y"}}]
    df, _ = _ingest(tmp_path, "a.json", lambda p: p.write_text(json.dumps(rows)))
    assert list(df.columns) == ["a", "meta.b"]


def test_jsonl_and_parquet(tmp_path):
    df, _ = _ingest(tmp_path, "a.jsonl", lambda p: p.write_text('{"a": 1}\n{"a": 2}\n'))
    assert df["a"].tolist() == [1, 2]
    df2, _ = _ingest(tmp_path, "b.parquet", lambda p: pd.DataFrame({"c": pd.Categorical(["u", "v"])}).to_parquet(p))
    assert df2["c"].tolist() == ["u", "v"]


@pytest.mark.parametrize("name,content,message", [
    ("a.pdf", b"%PDF", "Unsupported file type"),
    ("a.csv", b"", "empty"),
    ("a.csv", b"a,b\n", "no data rows"),
])
def test_bad_inputs_raise_user_facing_errors(tmp_path, name, content, message):
    src = tmp_path / name
    src.write_bytes(content)
    with pytest.raises(IngestError, match=message):
        ingest_file(src, tmp_path / "out.parquet")


def test_script_entrypoint(tmp_path, capsys):
    src = tmp_path / "a.csv"
    src.write_text("a,b\n1,2\n")
    assert main([str(src), str(tmp_path / "o.parquet")]) == 0
    assert json.loads(capsys.readouterr().out)["n_rows"] == 1
