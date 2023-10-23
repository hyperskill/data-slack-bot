from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any


# This class represents a simple database that stores its data as files in a directory.
class DB:
    def __init__(self, path: Path) -> None:
        """Initialize a DB instance."""
        self.path = path

    def get(self, key: str) -> str | None:
        """Get a value from the DB."""
        try:
            with (self.path / key).open(mode="rb") as f:
                return f.read().decode("utf-8")
        except FileNotFoundError:
            return None

    def __getitem__(self, key: str) -> str | None:
        return self.get(key)

    def write(self, key: str, value: Any) -> None:
        """Set a value in the DB."""
        with (self.path / key).open(mode="wb") as f:
            f.write(value)

    def delete(self, key: str) -> None:
        """Delete a value from the DB."""
        with contextlib.suppress(FileNotFoundError):
            Path.unlink(self.path / key)
