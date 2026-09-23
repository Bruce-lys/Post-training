"""Read-only observation helpers: log tailing, logging.jsonl metrics, GPU status, process state."""
from __future__ import annotations

import json
import os
import pathlib
import signal
import subprocess
import time
from typing import Any

import pandas as pd

from .constants import hostname

METRIC_FILE = "logging.jsonl"
LOG_FILE = "train.log"


# --------------------------------------------------------------------------- files
def tail_file(path: str | os.PathLike, offset: int = 0, max_bytes: int = 256_000) -> tuple[str, int]:
    """Read bytes after ``offset``; returns (text, new_offset). Handles truncation and missing files."""
    p = pathlib.Path(path)
    if not p.is_file():
        return "", 0
    size = p.stat().st_size
    if size < offset:
        offset = 0
    if size == offset:
        return "", offset
    with open(p, "rb") as fh:
        if size - offset > max_bytes:
            fh.seek(size - max_bytes)
            offset = size - max_bytes
        else:
            fh.seek(offset)
        data = fh.read()
    return data.decode("utf-8", "replace"), offset + len(data)


def read_tail(path: str | os.PathLike, max_bytes: int = 64_000) -> str:
    text, _ = tail_file(path, 0, max_bytes)
    # tqdm progress bars use \r; keep only the last segment of each line
    return "\n".join(line.rsplit("\r", 1)[-1] for line in text.splitlines())


def train_log_path(run_root: str) -> pathlib.Path:
    return pathlib.Path(run_root) / LOG_FILE


def _read_jsonl(path: pathlib.Path) -> list[dict]:
    rows: list[dict] = []
    if not path.is_file():
        return rows
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


# --------------------------------------------------------------------------- metrics
_METRIC_CACHE: dict[str, tuple[tuple, pd.DataFrame]] = {}


def _parse_iteration(v: Any) -> tuple[int | None, int | None]:
    if isinstance(v, str) and "/" in v:
        a, b = v.split("/", 1)
        try:
            return int(a), int(b)
        except ValueError:
            return None, None
    if isinstance(v, (int, float)):
        return int(v), None
    return None, None


def load_metrics(run_root: str) -> pd.DataFrame:
    """<run_root>/logging.jsonl -> DataFrame with a numeric ``step`` column (rows without metrics are skipped)."""
    f = pathlib.Path(run_root) / METRIC_FILE
    sig = (f.stat().st_mtime, f.stat().st_size) if f.is_file() else (0, 0)
    cached = _METRIC_CACHE.get(run_root)
    if cached and cached[0] == sig:
        return cached[1]
    rows = []
    for r in _read_jsonl(f):
        if not isinstance(r, dict) or not any(k in r for k in ("loss", "eval_loss")):
            continue
        step, total = _parse_iteration(r.get("iteration"))
        row = {k: v for k, v in r.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
        row["step"] = step if step is not None else len(rows) + 1
        if total is not None:
            row["total_steps"] = total
        for k in ("elapsed_time", "remaining_time"):
            if k in r:
                row[k] = r[k]
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("step").reset_index(drop=True)
    _METRIC_CACHE[run_root] = (sig, df)
    return df


def numeric_columns(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return []
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
            and c not in ("step", "total_steps")]


def progress(run_root: str) -> dict:
    """{'step', 'total', 'loss', 'remaining', 'elapsed'} from the latest metric row."""
    df = load_metrics(run_root) if run_root else pd.DataFrame()
    if df.empty:
        return {"step": 0, "total": None, "loss": None, "remaining": "", "elapsed": ""}
    last = df.iloc[-1]
    total = last.get("total_steps")
    return {"step": int(last["step"]), "total": int(total) if pd.notna(total) else None,
            "loss": float(last["loss"]) if "loss" in df.columns and pd.notna(last.get("loss")) else None,
            "remaining": str(last.get("remaining_time", "") or ""), "elapsed": str(last.get("elapsed_time", "") or "")}


def last_checkpoint(run_root: str) -> str:
    for r in reversed(_read_jsonl(pathlib.Path(run_root) / METRIC_FILE)):
        if isinstance(r, dict) and r.get("last_model_checkpoint"):
            return str(r["last_model_checkpoint"])
    return ""


# --------------------------------------------------------------------------- processes
def read_pid(run_root: str) -> int | None:
    p = pathlib.Path(run_root) / "ui" / "pid"
    if not p.is_file():
        return None
    try:
        return int(p.read_text().strip())
    except ValueError:
        return None


def proc_cmdline(pid: int) -> str:
    try:
        return pathlib.Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
    except OSError:
        return ""


def pid_alive(pid: int | None, run_root: str = "") -> bool:
    """True when ``pid`` exists and (when given) its command line mentions the run directory."""
    if not pid:
        return False
    cmd = proc_cmdline(pid)
    if not cmd:
        return False
    if run_root and run_root not in cmd:
        return False
    try:
        st = pathlib.Path(f"/proc/{pid}/stat").read_text().split(")")[-1].split()
        return st[0] != "Z"
    except (OSError, IndexError):
        return True


def exit_code(run_root: str) -> int | None:
    p = pathlib.Path(run_root) / "ui" / "exit_code"
    if not p.is_file():
        return None
    try:
        return int(p.read_text().strip())
    except ValueError:
        return None


def find_procs(needle: str, exclude_self: bool = True) -> list[int]:
    """PIDs whose command line contains ``needle`` (used to catch stray torchrun workers)."""
    out: list[int] = []
    me = os.getpid()
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        if exclude_self and pid == me:
            continue
        if needle in proc_cmdline(pid):
            out.append(pid)
    return out


def kill_group(pid: int, grace: float = 15.0) -> str:
    """SIGTERM the process group of ``pid``; SIGKILL survivors after ``grace`` seconds."""
    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError:
        return f"pid {pid} 已退出"
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return f"进程组 {pgid} 已退出"
    deadline = time.time() + grace
    while time.time() < deadline:
        if not pid_alive(pid):
            return f"进程组 {pgid} 已在 SIGTERM 后退出"
        time.sleep(0.5)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    time.sleep(1)
    return f"进程组 {pgid} 已 SIGKILL"


def run_status(doc: dict) -> dict:
    """Live status of a registered run: pid on the owning host, exit_code file elsewhere."""
    run_root = doc.get("run_root", "")
    local = doc.get("hostname") == hostname()
    pid = read_pid(run_root) if run_root else None
    code = exit_code(run_root) if run_root else None
    info: dict[str, Any] = {"local": local, "pid": pid, "exit_code": code, **progress(run_root)}
    if doc.get("status") == "aborted":
        info["status"] = "aborted"
    elif code is not None:
        info["status"] = "finished" if code == 0 else "failed"
    elif local:
        info["status"] = "running" if pid_alive(pid, run_root) else (
            doc.get("status") if doc.get("status") not in ("launching", "running") else "failed")
    else:
        info["status"] = doc.get("status", "unknown")
    return info


# --------------------------------------------------------------------------- host
def gpu_status() -> pd.DataFrame:
    cols = ["index", "name", "memory.used", "memory.total", "utilization.gpu", "temperature.gpu"]
    headers = ["GPU", "型号", "显存已用 MiB", "显存总量 MiB", "利用率 %", "温度 C"]
    try:
        r = subprocess.run(["nvidia-smi", f"--query-gpu={','.join(cols)}", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return pd.DataFrame(columns=headers)
    rows = [[x.strip() for x in line.split(",")] for line in r.stdout.strip().splitlines()]
    return pd.DataFrame([x for x in rows if len(x) == 6], columns=headers)


def gpu_memory_used_mib() -> list[int]:
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        return [int(x) for x in r.stdout.split() if x.strip().isdigit()]
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return []


def port_in_use(port: int) -> bool:
    try:
        r = subprocess.run(["ss", "-ltn"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return any(line.split()[3].endswith(f":{port}") for line in r.stdout.splitlines()[1:] if len(line.split()) > 3)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")
