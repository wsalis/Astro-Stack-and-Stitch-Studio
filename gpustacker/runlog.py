"""Small append-only run log for stack operations."""

from __future__ import annotations

import threading
import traceback
from datetime import datetime
from pathlib import Path
from types import TracebackType


class RunLog:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8", buffering=1)
        self._lock = threading.Lock()
        self._failed = False
        self.write("=" * 72)
        self.write("Run started")

    def write(self, message: str) -> None:
        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
        lines = str(message).splitlines() or [""]
        with self._lock:
            if self._failed or self._file.closed:
                return
            try:
                for line in lines:
                    self._file.write(f"{timestamp} {line}\n")
                self._file.flush()
            except OSError:
                self._failed = True
                try:
                    self._file.close()
                except OSError:
                    pass

    def write_exception(self, context: str, exc: BaseException) -> None:
        self.write(f"{context}: {type(exc).__name__}: {exc}")
        self.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))

    def close(self) -> None:
        if not self._file.closed:
            if not self._failed:
                self.write("Run ended")
            try:
                self._file.close()
            except OSError:
                self._failed = True

    def __enter__(self) -> RunLog:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback_info: TracebackType | None,
    ) -> bool:
        if exc is not None:
            self.write_exception("Run failed", exc)
        self.close()
        return False