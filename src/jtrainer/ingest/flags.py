"""Eligibility flags derived at ingest: tournament level and media dependence."""

from __future__ import annotations

import re
from collections.abc import Iterable


class Flagger:
    def __init__(self, tournament_patterns: Iterable[str], media_patterns: Iterable[str]):
        self._tournament = [re.compile(p, re.IGNORECASE) for p in tournament_patterns]
        self._media = [re.compile(p, re.IGNORECASE) for p in media_patterns]

    def is_tournament(self, notes: str | None) -> bool:
        return bool(notes) and any(p.search(notes) for p in self._tournament)

    def needs_media(self, clue_text: str, response: str = "") -> bool:
        # Responses like "[Points to his forehead]" are physical gestures.
        if response.lstrip().startswith("["):
            return True
        return any(p.search(clue_text) for p in self._media)
