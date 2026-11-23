"""LLM access: versioned prompts, a client interface, the Anthropic client, and a DB cache.

Every LLM result is cached in `llm_cache` keyed by (prompt_version, model, input hash);
the API is never called twice for the same input and prompt version.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from importlib import resources
from string import Template
from typing import Any, Protocol, TypeVar

import duckdb

from jtrainer.clock import Clock, SystemClock
from jtrainer.db import to_db

T = TypeVar("T")


class LLMUnavailable(RuntimeError):
    """No client configured, no credentials, or the API could not be reached."""


class MalformedResponse(ValueError):
    def __init__(self, message: str, raw: str):
        super().__init__(message)
        self.raw = raw


# ---------------------------------------------------------------------------
# Prompts


@dataclass(frozen=True)
class Prompt:
    name: str
    version: str
    system_template: str
    user_template: str

    def render(self, **fields: Any) -> tuple[str, str]:
        clean = {k: "" if v is None else str(v) for k, v in fields.items()}
        return (
            Template(self.system_template).substitute(clean),
            Template(self.user_template).substitute(clean),
        )


_FRONT_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)


def parse_prompt(name: str, text: str) -> Prompt:
    m = _FRONT_RE.match(text)
    meta: dict[str, str] = {}
    if m:
        for line in m.group(1).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
    version = meta.get("version")
    if not version:
        raise ValueError(f"prompt {name!r} has no version string in its front matter")
    body = text[m.end() :] if m else text
    if "===USER===" not in body:
        raise ValueError(f"prompt {name!r} has no ===USER=== section")
    system, user = body.split("===USER===", 1)
    return Prompt(name, version, system.strip(), user.strip())


def load_prompt(name: str) -> Prompt:
    text = (resources.files("jtrainer") / "prompts" / f"{name}.md").read_text()
    return parse_prompt(name, text)


# ---------------------------------------------------------------------------
# Requests and clients


@dataclass(frozen=True)
class LLMRequest:
    prompt_version: str
    model: str
    system: str
    user: str
    max_tokens: int = 2048
    schema: dict | None = None

    @property
    def input_hash(self) -> str:
        payload = json.dumps(
            {"system": self.system, "user": self.user, "schema": self.schema}, sort_keys=True
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def cache_key(self) -> str:
        raw = f"{self.prompt_version}|{self.model}|{self.input_hash}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass
class BatchResult:
    custom_id: str
    kind: str  # 'succeeded' | 'errored' | 'canceled' | 'expired'
    text: str | None = None
    error: str | None = None


class LLMClient(Protocol):
    def complete(self, request: LLMRequest) -> str: ...


class BatchClient(Protocol):
    def submit_batch(self, items: list[tuple[str, LLMRequest]]) -> str: ...

    def batch_status(self, batch_id: str) -> str: ...

    def batch_results(self, batch_id: str) -> Iterator[BatchResult]: ...


class AnthropicClient:
    """Real client. Structured JSON output via output_config.format."""

    def __init__(self, effort: str = "low", client: Any = None):
        self.effort = effort
        self._client = client

    @property
    def client(self):
        if self._client is None:
            import anthropic

            try:
                self._client = anthropic.Anthropic()
            except Exception as exc:  # missing credentials
                raise LLMUnavailable(str(exc)) from exc
        return self._client

    def _params(self, request: LLMRequest) -> dict:
        output_config: dict[str, Any] = {"effort": self.effort}
        if request.schema:
            output_config["format"] = {"type": "json_schema", "schema": request.schema}
        return {
            "model": request.model,
            "max_tokens": request.max_tokens,
            "system": [
                {"type": "text", "text": request.system, "cache_control": {"type": "ephemeral"}}
            ],
            "messages": [{"role": "user", "content": request.user}],
            "output_config": output_config,
        }

    @staticmethod
    def _text(message: Any) -> str:
        if getattr(message, "stop_reason", None) == "refusal":
            raise MalformedResponse("model refused", "")
        return "".join(b.text for b in message.content if getattr(b, "type", "") == "text")

    def complete(self, request: LLMRequest) -> str:
        import anthropic

        try:
            message = self.client.messages.create(**self._params(request))
        except (
            anthropic.APIConnectionError,
            anthropic.AuthenticationError,
            anthropic.RateLimitError,
            anthropic.InternalServerError,
        ) as exc:
            raise LLMUnavailable(str(exc)) from exc
        return self._text(message)

    def submit_batch(self, items: list[tuple[str, LLMRequest]]) -> str:
        batch = self.client.messages.batches.create(
            requests=[{"custom_id": cid, "params": self._params(req)} for cid, req in items]
        )
        return batch.id

    def batch_status(self, batch_id: str) -> str:
        return self.client.messages.batches.retrieve(batch_id).processing_status

    def batch_results(self, batch_id: str) -> Iterator[BatchResult]:
        for res in self.client.messages.batches.results(batch_id):
            kind = res.result.type
            if kind == "succeeded":
                try:
                    yield BatchResult(res.custom_id, kind, text=self._text(res.result.message))
                except MalformedResponse as exc:
                    yield BatchResult(res.custom_id, "errored", error=str(exc))
            elif kind == "errored":
                yield BatchResult(res.custom_id, kind, error=str(res.result.error))
            else:
                yield BatchResult(res.custom_id, kind)


class NoLLM:
    """Stand-in when the API is not configured; every call raises LLMUnavailable."""

    def complete(self, request: LLMRequest) -> str:
        raise LLMUnavailable("no LLM client configured (set ANTHROPIC_API_KEY)")


def default_client(effort: str = "low") -> LLMClient:
    import os

    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return AnthropicClient(effort=effort)
    return NoLLM()


# ---------------------------------------------------------------------------
# Cache


def parse_json_object(text: str) -> dict:
    """Parse a JSON object, tolerating surrounding whitespace or a ```json fence."""
    s = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", s, re.DOTALL)
    if fence:
        s = fence.group(1)
    try:
        obj = json.loads(s)
    except json.JSONDecodeError as exc:
        raise MalformedResponse(f"invalid JSON: {exc}", text) from exc
    if not isinstance(obj, dict):
        raise MalformedResponse("expected a JSON object", text)
    return obj


class CachedLLM:
    def __init__(
        self, client: LLMClient, conn: duckdb.DuckDBPyConnection, clock: Clock | None = None
    ):
        self.client = client
        self.conn = conn
        self.clock = clock or SystemClock()

    def cached_text(self, request: LLMRequest) -> str | None:
        row = self.conn.execute(
            "SELECT response_text FROM llm_cache WHERE cache_key = ?", [request.cache_key]
        ).fetchone()
        return row[0] if row else None

    def store(self, request: LLMRequest, text: str) -> None:
        self.conn.execute(
            "INSERT INTO llm_cache VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT DO NOTHING",
            [
                request.cache_key,
                request.prompt_version,
                request.model,
                request.input_hash,
                text,
                to_db(self.clock.now()),
            ],
        )

    def complete(self, request: LLMRequest, parse: Callable[[str], T], retries: int = 1) -> T:
        """Return the parsed result, from cache when possible.

        Malformed output is retried `retries` times, then MalformedResponse propagates.
        Only output that parses is cached.
        """
        cached = self.cached_text(request)
        if cached is not None:
            return parse(cached)
        last: MalformedResponse | None = None
        for _ in range(retries + 1):
            text = self.client.complete(request)
            try:
                result = parse(text)
            except MalformedResponse as exc:
                last = exc
                continue
            self.store(request, text)
            return result
        assert last is not None
        raise last
