"""Injectable clocks. All application time goes through a Clock."""

from __future__ import annotations

import datetime as dt
import time
from typing import Protocol

UTC = dt.UTC


class Clock(Protocol):
    def now(self) -> dt.datetime: ...

    def monotonic_ms(self) -> int: ...

    def sleep(self, seconds: float) -> None: ...


class SystemClock:
    def now(self) -> dt.datetime:
        return dt.datetime.now(UTC)

    def monotonic_ms(self) -> int:
        return int(time.monotonic() * 1000)

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


class FakeClock:
    """Deterministic clock. `sleep` advances time instead of blocking."""

    def __init__(self, start: dt.datetime | None = None):
        self._now = start or dt.datetime(2026, 1, 1, tzinfo=UTC)
        self._mono = 0

    def now(self) -> dt.datetime:
        return self._now

    def monotonic_ms(self) -> int:
        return self._mono

    def advance(self, ms: float) -> None:
        self._now += dt.timedelta(milliseconds=ms)
        self._mono += int(ms)

    def sleep(self, seconds: float) -> None:
        self.advance(seconds * 1000)
