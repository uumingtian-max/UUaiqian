"""Server-side shell execution for user-requested deploy steps (private instance)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

_DEFAULT_CWD = os.environ.get("SIHAN_DEPLOY_CWD", "/home/linux/sihan-final")
_DEFAULT_TIMEOUT = int(os.environ.get("SIHAN_DEPLOY_TIMEOUT", "180"))


def bash_syntax_check(script: str) -> tuple[bool, str]:
    """bash -n 语法检查，不执行。"""
    s = (script or "").strip()
    if not s:
        return False, "empty script"
    import tempfile

    path = ""
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False, encoding="utf-8") as tf:
            tf.write(s)
            path = tf.name
        proc = subprocess.run(
            ["bash", "-n", path],
            capture_output=True,
            text=True,
            timeout=15,
        )
        err = (proc.stderr or "").strip() or (proc.stdout or "").strip()
        if proc.returncode != 0:
            return False, err or f"bash -n exit {proc.returncode}"
        return True, ""
    except subprocess.TimeoutExpired:
        return False, "bash -n timeout"
    except OSError as e:
        return False, str(e)
    finally:
        if path:
            try:
                os.unlink(path)
            except OSError:
                pass


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
