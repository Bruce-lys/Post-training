"""Read schema/megatron_sft_args.json (exported from ms-swift's MegatronSftArguments by scripts/export_schema.sh).

The UI process never imports swift/torch; it only consults this JSON for defaults, choices, help text and
the set of known flags (used to sanity-check the free-form advanced arguments)."""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from .constants import SCHEMA_FILE


@lru_cache(maxsize=1)
def load() -> dict[str, dict]:
    if not SCHEMA_FILE.is_file():
        return {}
    try:
        return json.loads(SCHEMA_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def known() -> bool:
    return bool(load())


def field(name: str) -> dict | None:
    return load().get(name)


def default(name: str, fallback: Any = None) -> Any:
    f = field(name)
    return fallback if f is None else f.get("default", fallback)


def choices(name: str) -> list | None:
    f = field(name)
    if not f or not f.get("choices"):
        return None
    return [c for c in f["choices"] if c is not None]


def help_text(name: str) -> str:
    f = field(name)
    return (f or {}).get("help") or ""


def is_known_flag(flag: str) -> bool:
    """Is ``--foo`` a MegatronSftArguments field? When the schema is missing, assume yes."""
    if not known():
        return True
    return flag.lstrip("-") in load()
