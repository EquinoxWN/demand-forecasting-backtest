"""Atomic file writes: readers see the old file or the complete new one, never a partial one."""

from __future__ import annotations

import gzip
import io
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO


@contextmanager
def atomic_text(path: Path) -> Iterator[IO[str]]:
    """Write text to a temporary sibling, then rename it over path (gzip for .gz)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        if path.suffix == ".gz":
            # mtime=0 keeps the gzip header stable, so the same data gives the same bytes.
            with (
                gzip.GzipFile(tmp, "wb", mtime=0) as raw,
                io.TextIOWrapper(raw, encoding="utf-8", newline="") as fh,
            ):
                yield fh
        else:
            with tmp.open("w", encoding="utf-8", newline="") as fh:
                yield fh
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def write_text(path: Path, text: str) -> None:
    """Atomically replace path with text."""
    with atomic_text(path) as fh:
        fh.write(text)
