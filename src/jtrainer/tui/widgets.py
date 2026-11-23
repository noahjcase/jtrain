"""Shared widgets."""

from __future__ import annotations

from textual import events
from textual.widgets import Input


class AnswerInput(Input):
    """An Input that cooperates with its screen's answer flow.

    - Enter is handled synchronously by the screen's `submit_text` instead of through a
      posted Submitted message, so the screen's state changes before the next key is read.
    - Every key is first offered to the screen's `intercept_key`, so a confidence key
      typed straight after Enter reaches the confidence prompt wherever focus is.

    Together these make the Enter -> confidence ordering explicit rather than dependent on
    message-queue timing.
    """

    async def _on_key(self, event: events.Key) -> None:
        intercept = getattr(self.screen, "intercept_key", None)
        if intercept is not None and intercept(event.key):
            event.stop()
            event.prevent_default()
            return
        await super()._on_key(event)

    async def action_submit(self) -> None:
        handler = getattr(self.screen, "submit_text", None)
        if handler is None:
            await super().action_submit()
        else:
            handler(self.value)
