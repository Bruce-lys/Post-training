"""Run registry on the shared filesystem: one JSON per run + an append-only event log."""
from __future__ import annotations

import json
import os
import time
from typing import Any

from .constants import EVENTS_FILE, RUNS_DIR


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _path(run_id: str):
    return RUNS_DIR / f"{run_id}.json"


def _write(path, doc: dict) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(doc, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def event(kind: str, run_id: str, **extra: Any) -> None:
    EVENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    row = {"ts": _now(), "event": kind, "run_id": run_id, **extra}
    with open(EVENTS_FILE, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def register(doc: dict) -> dict:
    doc = {"created_at": _now(), "status": "launching", **doc}
    _write(_path(doc["run_id"]), doc)
    event("register", doc["run_id"], status=doc["status"], host=doc.get("hostname"))
    return doc


def get(run_id: str) -> dict | None:
    p = _path(run_id)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def update(run_id: str, **fields: Any) -> dict | None:
    doc = get(run_id)
    if doc is None:
        return None
    doc.update(fields)
    doc["updated_at"] = _now()
    _write(_path(run_id), doc)
    if "status" in fields:
        event("status", run_id, status=fields["status"])
    return doc


def delete(run_id: str) -> bool:
    p = _path(run_id)
    if p.is_file():
        p.unlink()
        event("delete", run_id)
        return True
    return False


def list_runs(limit: int = 500) -> list[dict]:
    if not RUNS_DIR.is_dir():
        return []
    docs = []
    for p in RUNS_DIR.glob("*.json"):
        try:
            docs.append(json.loads(p.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    docs.sort(key=lambda d: d.get("created_at", ""), reverse=True)
    return docs[:limit]


def active_runs(host: str | None = None) -> list[dict]:
    return [d for d in list_runs() if d.get("status") in ("launching", "running")
            and (host is None or d.get("hostname") == host)]
