from __future__ import annotations

import logging


class NoThreadError(Exception):
    """Raised when a thread is not found."""

    def __init__(self, message: str = "Thread does not exist.") -> None:
        super().__init__(message)
        logging.error(message)


class NoRunError(Exception):
    """Raised when a run is not found."""

    def __init__(self, message: str = "Run does not exist.") -> None:
        super().__init__(message)
        logging.error(message)


class TimeIsUpError(Exception):
    """Raised when time is up."""

    def __init__(self, message: str = "Time is up!") -> None:
        super().__init__(message)
        logging.error(message)
