"""Prepares a table for forecasting: one regular series per id.

Nothing is assumed silently:
- Duplicate (series, timestamp) pairs stop the run: whether to sum or average
  them is a business decision the user has to make.
- The frequency is inferred from the timestamps; if fewer than 90% of steps
  match it, the data is irregular and the run stops.
- Missing steps are linearly interpolated only when they are at most 10% of a
  series, and the count is reported. More than that stops the run.
- Unparseable dates stop the run if above 1% of rows; otherwise they are dropped
  and reported.

Writes the engineered dataset as (series_id, timestamp, value) and scores the
naive and seasonal-naive baselines with the same backtest the models get.
"""
import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from app import storage
from app.training import tsmetrics

MAX_MISSING_SHARE = 0.10
MAX_BAD_DATES_SHARE = 0.01
MIN_REGULAR_SHARE = 0.90
DEFAULT_WINDOWS = 3

# offset class -> (seasonal period, default horizon)
_SEASONALITY = {
    "Minute": (60, 60), "Hour": (24, 24), "BusinessHour": (8, 8), "Day": (7, 14), "BusinessDay": (5, 10),
    "Week": (52, 8), "MonthBegin": (12, 6), "MonthEnd": (12, 6), "QuarterBegin": (4, 4),
    "QuarterEnd": (4, 4), "YearBegin": (1, 2), "YearEnd": (1, 2),
}


class TimeSeriesError(ValueError):
    """The data can't be forecast as-is. The message is safe to show the user."""


def infer_frequency(frame: pd.DataFrame) -> str:
    """Frequency alias shared by the series (e.g. 'D', 'h', 'MS')."""
    for _, g in sorted(frame.groupby("series_id"), key=lambda kv: -len(kv[1])):
        if len(g) >= 3:
            freq = pd.infer_freq(pd.DatetimeIndex(g["timestamp"]))
            if freq:
                return freq
    diffs = frame.groupby("series_id")["timestamp"].diff().dropna()
    if diffs.empty:
        raise TimeSeriesError("Each series needs at least 3 timestamps to infer its frequency.")
    step = diffs.mode().iloc[0]
    if (diffs == step).mean() < MIN_REGULAR_SHARE:
        raise TimeSeriesError(
            f"Timestamps are irregular: only {100 * (diffs == step).mean():.0f}% of steps equal the most common "
            f"spacing ({step}). Resample to a fixed frequency first."
        )
    return pd.tseries.frequencies.to_offset(step).freqstr


def season_and_horizon(freq: str) -> tuple[int, int]:
    offset = pd.tseries.frequencies.to_offset(freq)
    season, horizon = _SEASONALITY.get(type(offset).__name__, (1, 12))
    return season, horizon


def prepare(run_id: str, date_column: str, target: str, id_column: str | None = None,
            horizon: int | None = None) -> dict:
    df = storage.load_dataset(run_id)
    for col in [date_column, target] + ([id_column] if id_column else []):
        if col not in df.columns:
            raise TimeSeriesError(f"Column {col!r} not found.")
    if not is_numeric_dtype(df[target]) or is_bool_dtype(df[target]):
        raise TimeSeriesError(f"Forecast target {target!r} must be numeric.")

    frame = pd.DataFrame({
        "series_id": df[id_column].astype(str) if id_column else "series",
        "timestamp": pd.to_datetime(df[date_column], errors="coerce", format="mixed"),
        "value": df[target].astype(float),
    })
    bad = int(frame["timestamp"].isna().sum())
    actions = []
    if bad:
        if bad / len(frame) > MAX_BAD_DATES_SHARE:
            raise TimeSeriesError(f"{bad} of {len(frame)} values in {date_column!r} are not dates.")
        frame = frame[frame["timestamp"].notna()]
        actions.append(f"Dropped {bad} row(s) whose {date_column!r} is not a date.")
    if getattr(frame["timestamp"].dt, "tz", None) is not None:
        frame["timestamp"] = frame["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)
        actions.append("Converted timezone-aware timestamps to UTC.")

    dupes = int(frame.duplicated(["series_id", "timestamp"]).sum())
    if dupes:
        hint = "" if id_column else " If the file holds several series, choose the column that identifies them."
        raise TimeSeriesError(
            f"{dupes} rows repeat a timestamp within the same series. Aggregate them first (sum or average "
            f"is your call).{hint}"
        )

    frame = frame.sort_values(["series_id", "timestamp"])
    freq = infer_frequency(frame)
    season, default_horizon = season_and_horizon(freq)
    H = int(horizon or default_horizon)

    parts, filled = [], {}
    for sid, g in frame.groupby("series_id", sort=True):
        full = pd.date_range(g["timestamp"].iloc[0], g["timestamp"].iloc[-1], freq=freq)
        s = g.set_index("timestamp")["value"].reindex(full)
        off_grid = len(g) - int(g["timestamp"].isin(full).sum())
        if off_grid:
            raise TimeSeriesError(f"Series {sid!r} has {off_grid} timestamp(s) that don't fall on the {freq} grid.")
        missing = int(s.isna().sum())
        if missing / len(s) > MAX_MISSING_SHARE:
            raise TimeSeriesError(
                f"Series {sid!r} is missing {missing} of {len(s)} steps ({100 * missing / len(s):.0f}%); "
                f"at most {100 * MAX_MISSING_SHARE:.0f}% can be interpolated."
            )
        if missing:
            s = s.interpolate(method="linear", limit_direction="both")
            filled[str(sid)] = missing
        parts.append(pd.DataFrame({"series_id": str(sid), "timestamp": s.index, "value": s.to_numpy()}))
    series = pd.concat(parts, ignore_index=True)
    if filled:
        actions.append(f"Interpolated {sum(filled.values())} missing step(s) across {len(filled)} series.")

    shortest = int(series.groupby("series_id").size().min())
    L = int(max(2 * season, 2 * H, 16))
    windows = DEFAULT_WINDOWS
    while windows > 1 and shortest < H * (windows + 1) + H:
        windows -= 1
    if shortest < 2 * H + 2:
        raise TimeSeriesError(
            f"The shortest series has {shortest} points; forecasting {H} steps needs at least {2 * H + 2} "
            "(one horizon to learn from and one to test on). Choose a shorter horizon."
        )
    L = int(min(L, max(4, shortest - H * (windows + 1))))

    storage.save_engineered(run_id, series)
    report = {
        "date_column": date_column, "target": target, "id_column": id_column,
        "freq": freq, "season": season, "horizon": H, "input_length": L, "windows": windows,
        "n_series": int(series["series_id"].nunique()), "n_points": int(len(series)),
        "shortest_series": shortest, "start": str(series["timestamp"].min()), "end": str(series["timestamp"].max()),
        "interpolated_steps": filled, "actions": actions,
        "baselines": baselines(series, H, season, windows),
    }
    storage.write_json(run_id, "timeseries.json", report)
    return report


def baselines(series: pd.DataFrame, H: int, season: int, windows: int) -> dict:
    """Naive and seasonal-naive forecasts, scored with the models' backtest."""
    evals = {"Naive (last value)": [], "Seasonal naive": []}
    for _, g in series.groupby("series_id"):
        v = g["value"].to_numpy(dtype=np.float64)
        for cut in tsmetrics.backtest_cuts(len(v), H, windows):
            if cut < H + 2:
                continue
            actual, history = v[cut:cut + H], v[:cut]
            evals["Naive (last value)"].append(tsmetrics.evaluate(actual, tsmetrics.naive(history, H), history, season))
            evals["Seasonal naive"].append(
                tsmetrics.evaluate(actual, tsmetrics.seasonal_naive(history, H, season), history, season))
    return {name: tsmetrics.aggregate(e) for name, e in evals.items() if e}
