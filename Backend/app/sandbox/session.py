"""Host side of the code sandbox: starts app/sandbox/driver.py somewhere
isolated and exchanges code cells with it over stdin/stdout.

Security model for the Modal backend (production):
- gVisor-isolated container, created per run and terminated after it.
- block_network=True: code can't download anything or exfiltrate data.
- No secrets and no volumes: the only data inside is this run's dataset,
  pushed in by the (trusted) host. The sandbox never receives a URL or a
  credential for fetching it.
- CPU, memory, per-cell, and total-lifetime limits.
- Nothing produced inside is unpickled or executed on the host; only JSON text
  comes back.
"""
import json
import logging
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from app.config import settings

log = logging.getLogger(__name__)

DRIVER_SOURCE_PATH = Path(__file__).with_name("driver.py")
REMOTE_DATA_PATH = "/sandbox/data/dataset.parquet"
REMOTE_DRIVER_PATH = "/sandbox/driver.py"

# Extra wall time the host allows past the in-sandbox cell timeout before it
# kills the whole session (covers code stuck inside C extensions, which the
# in-sandbox alarm can't interrupt).
DEFAULT_KILL_GRACE_SECONDS = 15
STARTUP_TIMEOUT_SECONDS = 180  # first Modal run builds the image


@dataclass
class ExecResult:
    ok: bool
    stdout: str
    error: str | None
    seconds: float
    truncated: bool = False
    session_reset: bool = False  # the session died and was restarted: variables are gone

    def as_tool_output(self) -> str:
        parts = []
        if self.session_reset:
            parts.append("[session was restarted: earlier variables are gone; `df` was reloaded]")
        if self.stdout:
            parts.append(self.stdout.rstrip())
        if self.error:
            parts.append("ERROR:\n" + self.error.rstrip())
        if not parts:
            parts.append("(no output — print() what you need to see)")
        return "\n".join(parts)


class SandboxError(RuntimeError):
    pass


class _DriverSession:
    """Protocol handling shared by every backend. Subclasses implement
    _spawn() -> (write(str), iterator of stdout text chunks) and _kill()."""

    def __init__(self, dataset_path: Path, kill_grace_seconds: int = DEFAULT_KILL_GRACE_SECONDS):
        self._dataset_path = dataset_path
        self._kill_grace = kill_grace_seconds
        self._marker = f"@@NAMTHEG-{uuid.uuid4().hex}@@"
        self._lines: queue.Queue = queue.Queue()
        self._write = None
        self._next_id = 1
        self._alive = False

    # -- backend hooks -------------------------------------------------------
    def _spawn(self):
        raise NotImplementedError

    def _kill(self) -> None:
        raise NotImplementedError

    # -- lifecycle -----------------------------------------------------------
    def start(self) -> "_DriverSession":
        self._lines = queue.Queue()
        self._write, chunks = self._spawn()
        threading.Thread(target=self._pump, args=(chunks, self._lines), daemon=True).start()
        ready = self._await(0, time.monotonic() + STARTUP_TIMEOUT_SECONDS)
        if ready is None or not ready.get("ok"):
            detail = (ready or {}).get("error") or "no response from sandbox driver"
            self.close()
            raise SandboxError(f"Sandbox failed to start: {detail}")
        self._alive = True
        return self

    def close(self) -> None:
        self._alive = False
        try:
            self._kill()
        except Exception as e:
            log.warning("Error while closing sandbox: %s", e)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- I/O -----------------------------------------------------------------
    def _pump(self, chunks, lines: queue.Queue) -> None:
        """Reassemble stdout into lines regardless of how the backend chunks it."""
        buf = ""
        try:
            for chunk in chunks:
                buf += chunk if isinstance(chunk, str) else chunk.decode("utf-8", errors="replace")
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    lines.put(line)
        except Exception as e:
            log.debug("Sandbox stdout reader stopped: %s", e)
        lines.put(None)  # EOF

    def _await(self, request_id: int, deadline: float) -> dict | None:
        """Next protocol message with this id; None on EOF or deadline."""
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            try:
                line = self._lines.get(timeout=remaining)
            except queue.Empty:
                return None
            if line is None:
                return None
            if not line.startswith(self._marker):
                continue  # stray output (import warnings etc.)
            try:
                msg = json.loads(line[len(self._marker):])
            except json.JSONDecodeError:
                continue
            if msg.get("id") == request_id:
                return msg

    def run(self, code: str, timeout: int | None = None) -> ExecResult:
        timeout = int(timeout or settings.sandbox_exec_timeout_seconds)
        reset = False
        if not self._alive:
            self.start()
            reset = True
        request_id = self._next_id
        self._next_id += 1
        started = time.monotonic()
        try:
            self._write(json.dumps({"id": request_id, "code": code, "timeout": timeout}) + "\n")
        except Exception as e:
            self.close()
            return ExecResult(False, "", f"Sandbox connection lost: {e}", 0.0, session_reset=True)

        msg = self._await(request_id, started + timeout + self._kill_grace)
        if msg is None:
            elapsed = time.monotonic() - started
            timed_out = elapsed >= timeout
            self.close()
            reason = (
                f"Cell exceeded {timeout}s and the session was killed."
                if timed_out
                else "The sandbox process died (most likely out of memory)."
            )
            return ExecResult(False, "", reason + " Variables are gone; `df` will be reloaded on the next run.",
                              round(elapsed, 3), session_reset=True)
        return ExecResult(
            ok=bool(msg.get("ok")),
            stdout=msg.get("stdout") or "",
            error=msg.get("error"),
            seconds=float(msg.get("seconds") or 0.0),
            truncated=bool(msg.get("truncated")),
            session_reset=reset,
        )


class LocalSandbox(_DriverSession):
    """Runs the driver as a plain subprocess on this machine.

    NOT ISOLATED: the code can read files and reach the network. For local
    development and tests only; enable with SANDBOX_BACKEND=local.
    """

    def __init__(self, dataset_path: Path, **kw):
        super().__init__(dataset_path, **kw)
        self._proc: subprocess.Popen | None = None
        self._workdir = tempfile.mkdtemp(prefix="namtheg-sandbox-")

    def _spawn(self):
        # Keep credentials out of the child even in dev.
        env = {k: v for k, v in os.environ.items()
               if not any(s in k.upper() for s in ("KEY", "SECRET", "TOKEN", "PASSWORD"))}
        self._proc = subprocess.Popen(
            [sys.executable, "-u", str(DRIVER_SOURCE_PATH), str(self._dataset_path.resolve()), self._marker],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            cwd=self._workdir,
            env=env,
        )
        proc = self._proc

        def write(data: str) -> None:
            proc.stdin.write(data)
            proc.stdin.flush()

        return write, iter(proc.stdout.readline, "")

    def _kill(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.kill()
            self._proc.wait(timeout=5)


def _modal_image():
    import modal

    return modal.Image.debian_slim(python_version="3.11").pip_install(
        "pandas>=2.2,<3.0",
        "numpy>=1.26,<3.0",
        "pyarrow>=17,<22",
        "scipy>=1.11,<2.0",
        "scikit-learn>=1.5,<2.0",
        "statsmodels>=0.14,<1.0",
    )


class ModalSandbox(_DriverSession):
    """Modal Sandbox (gVisor): the production backend. See module docstring."""

    def __init__(self, dataset_path: Path, **kw):
        super().__init__(dataset_path, **kw)
        self._sb = None

    def _spawn(self):
        import modal

        if self._sb is None:
            app = modal.App.lookup(settings.sandbox_app_name, create_if_missing=True)
            self._sb = modal.Sandbox.create(
                app=app,
                image=_modal_image(),
                timeout=settings.sandbox_lifetime_seconds,
                cpu=settings.sandbox_cpu,
                memory=settings.sandbox_memory_mb,
                block_network=True,
            )
            self._sb.filesystem.write_bytes(self._dataset_path.read_bytes(), REMOTE_DATA_PATH)
            self._sb.filesystem.write_text(DRIVER_SOURCE_PATH.read_text(encoding="utf-8"), REMOTE_DRIVER_PATH)
        proc = self._sb.exec(
            "python", "-u", REMOTE_DRIVER_PATH, REMOTE_DATA_PATH, self._marker,
            stderr=modal.stream_type.StreamType.DEVNULL,
            text=True,
        )

        def write(data: str) -> None:
            proc.stdin.write(data.encode("utf-8"))
            proc.stdin.drain()

        return write, iter(proc.stdout)

    def _kill(self) -> None:
        # Killing only the driver would leave a runaway cell burning CPU, so
        # the whole sandbox goes; start() creates a fresh one if needed.
        if self._sb is not None:
            sb, self._sb = self._sb, None
            sb.terminate()


def open_sandbox(dataset_path: Path) -> _DriverSession | None:
    """Start the configured sandbox with this dataset loaded as `df`.
    Returns None when SANDBOX_BACKEND=none. Raises SandboxError/others on failure."""
    backend = settings.sandbox_backend.strip().lower()
    if backend == "none":
        return None
    if backend == "local":
        log.warning("SANDBOX_BACKEND=local: agent code runs UNISOLATED on this machine. Dev/test only.")
        return LocalSandbox(dataset_path).start()
    if backend == "modal":
        return ModalSandbox(dataset_path).start()
    raise SandboxError(f"Unknown SANDBOX_BACKEND {settings.sandbox_backend!r} (expected modal, local or none).")
