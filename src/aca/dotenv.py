"""Minimal ``.env`` loader (stdlib only), so provider API keys can live in a file.

Credentials are read from ``<data-dir>/.env`` (next to ``config.json``) and a ``.env`` in the
current directory, and set into ``os.environ`` — the provider factory reads ``os.environ``, so it
needs no change. The real shell environment always wins: a ``.env`` never overrides an explicit
export, and an earlier file wins over a later one for the same key. Only ``KEY=VALUE`` lines are
parsed (optional ``export`` prefix, ``#`` comments, single/double-quoted values); this is just
enough for credentials, not a full dotenv implementation. Values are set silently — never logged.
"""

from __future__ import annotations

import os
from pathlib import Path


def parse_env(text: str) -> dict[str, str]:
    """Parse ``.env`` text into a dict of ``KEY -> value``."""
    result: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, value = line.partition("=")
        key = key.strip()
        if not sep or not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]  # strip matching surrounding quotes
        result[key] = value
    return result


def load_dotenv(*paths: Path, override: bool = False) -> list[Path]:
    """Load each existing ``.env`` in ``paths`` into ``os.environ``; return the files applied.

    Without ``override`` (the default) an existing environment variable is left untouched, so a
    shell export beats a ``.env`` and the first file listed beats later ones for the same key.
    """
    applied: list[Path] = []
    for path in paths:
        try:
            if not path.is_file():
                continue
            data = parse_env(path.read_text(encoding="utf-8"))
        except OSError:
            continue
        for key, value in data.items():
            if override or key not in os.environ:
                os.environ[key] = value
        applied.append(path)
    return applied
