import pandas as pd
import pytest

from app.sandbox import LocalSandbox


@pytest.fixture
def sandbox(tmp_path):
    path = tmp_path / "d.parquet"
    pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "x"]}).to_parquet(path)
    with LocalSandbox(path, kill_grace_seconds=1).start() as sb:
        yield sb


def test_df_is_preloaded_and_state_persists(sandbox):
    assert sandbox.run("total = df['a'].sum()").ok
    r = sandbox.run("total * 2")
    assert r.ok and r.stdout.strip() == "np.int64(12)"


def test_errors_return_traceback_and_session_survives(sandbox):
    r = sandbox.run("df['missing']")
    assert not r.ok and "KeyError" in r.error
    for code in ("exit()", "import sys; sys.exit(3)", "input()"):
        assert not sandbox.run(code).ok
    r = sandbox.run("print(len(df))")
    assert r.ok and r.stdout.strip() == "3" and not r.session_reset


def test_runaway_cell_is_killed_and_session_restarts(sandbox):
    sandbox.run("marker = 1")
    r = sandbox.run("while True: pass", timeout=1)
    assert not r.ok and r.session_reset
    r = sandbox.run("print('marker' in globals(), df.shape)")
    assert r.ok and r.session_reset and "False (3, 2)" in r.stdout


def test_output_is_truncated(sandbox):
    r = sandbox.run("print('z' * 50_000)")
    assert r.ok and r.truncated and len(r.stdout) < 25_000


def test_cells_cannot_see_host_secrets(sandbox):
    r = sandbox.run("import os; print(sorted(k for k in os.environ if 'KEY' in k.upper() or 'SECRET' in k.upper()))")
    assert r.stdout.strip() == "[]"
