"""Server-side shell execution for user-requested deploy steps (private instance)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

_DEFAULT_CWD = os.environ.get("SIHAN_DEPLOY_CWD", "/home/linux/sihan-final")
_DEFAULT_TIMEOUT = int(os.environ.get("SIHAN_DEPLOY_TIMEOUT", "180"))


def resolve_working_dir(cwd: str | None) -> Path:
    if cwd and str(cwd).strip():
        p = Path(str(cwd).strip()).expanduser().resolve()
        if p.is_dir():
            return p
    p = Path(_DEFAULT_CWD).expanduser().resolve()
    if p.is_dir():
        return p
    return Path("/tmp")


def run_shell(command: str, cwd: str | None = None, timeout_sec: int | None = None) -> dict[str, object]:
    """Run a shell command; returns stdout, stderr, returncode. No command allowlist."""
    cmd = (command or "").strip()
    if not cmd:
        return {"ok": False, "error": "empty command", "returncode": -1, "stdout": "", "stderr": ""}
    timeout = timeout_sec if timeout_sec is not None else _DEFAULT_TIMEOUT
    work = resolve_working_dir(cwd)
    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            cwd=str(work),
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ},
        )
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout or "",
            "stderr": proc.stderr or "",
            "cwd": str(work),
        }
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "returncode": -124,
            "stdout": "",
            "stderr": f"timeout after {timeout}s",
            "cwd": str(work),
        }
    except Exception as e:  # pragma: no cover
        return {
            "ok": False,
            "returncode": -1,
            "stdout": "",
            "stderr": str(e),
            "cwd": str(work),
        }
