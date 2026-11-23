"""The real Anthropic client wrapper, against a stub SDK object (no network)."""

from types import SimpleNamespace

import pytest

from jtrainer.llm import (
    AnthropicClient,
    CachedLLM,
    LLMRequest,
    LLMUnavailable,
    MalformedResponse,
    NoLLM,
    default_client,
    load_prompt,
    parse_json_object,
)


def text_message(text, stop="end_turn"):
    return SimpleNamespace(
        stop_reason=stop,
        content=[
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text=text),
        ],
    )


class StubMessages:
    def __init__(self):
        self.created = []
        self.batches = SimpleNamespace(
            create=self._batch_create, retrieve=self._retrieve, results=self._results
        )
        self.batch_requests = None

    def create(self, **params):
        self.created.append(params)
        return text_message('{"ok": true}')

    def _batch_create(self, requests):
        self.batch_requests = requests
        return SimpleNamespace(id="msgbatch_1")

    def _retrieve(self, batch_id):
        return SimpleNamespace(processing_status="ended")

    def _results(self, batch_id):
        yield SimpleNamespace(
            custom_id="clue-2",
            result=SimpleNamespace(type="succeeded", message=text_message('{"a": 1}')),
        )
        yield SimpleNamespace(
            custom_id="clue-1",
            result=SimpleNamespace(type="errored", error={"type": "overloaded_error"}),
        )
        yield SimpleNamespace(custom_id="clue-3", result=SimpleNamespace(type="expired"))
        yield SimpleNamespace(
            custom_id="clue-4",
            result=SimpleNamespace(type="succeeded", message=text_message("", stop="refusal")),
        )


REQ = LLMRequest(
    "judge-v1",
    "test-model",
    "system text",
    "user text",
    512,
    {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
)


def test_params_shape_structured_output_and_caching():
    stub = SimpleNamespace(messages=StubMessages())
    client = AnthropicClient(effort="low", client=stub)
    assert client.complete(REQ) == '{"ok": true}'
    params = stub.messages.created[0]
    assert params["model"] == "test-model" and params["max_tokens"] == 512
    assert params["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert params["output_config"]["effort"] == "low"
    assert params["output_config"]["format"]["type"] == "json_schema"
    assert params["messages"] == [{"role": "user", "content": "user text"}]
    assert "thinking" not in params and "temperature" not in params


def test_batch_submit_and_results_by_custom_id():
    stub = SimpleNamespace(messages=StubMessages())
    client = AnthropicClient(client=stub)
    assert client.submit_batch([("clue-2", REQ)]) == "msgbatch_1"
    assert stub.messages.batch_requests[0]["custom_id"] == "clue-2"
    assert client.batch_status("msgbatch_1") == "ended"
    results = {r.custom_id: r for r in client.batch_results("msgbatch_1")}
    assert results["clue-2"].kind == "succeeded" and results["clue-2"].text == '{"a": 1}'
    assert results["clue-1"].kind == "errored"
    assert results["clue-3"].kind == "expired"
    assert results["clue-4"].kind == "errored"  # a refusal is not a usable result


def test_refusal_is_malformed():
    class Refuse(StubMessages):
        def create(self, **params):
            return text_message("", stop="refusal")

    client = AnthropicClient(client=SimpleNamespace(messages=Refuse()))
    with pytest.raises(MalformedResponse):
        client.complete(REQ)


def test_no_llm_and_default_client(monkeypatch):
    with pytest.raises(LLMUnavailable):
        NoLLM().complete(REQ)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    assert isinstance(default_client(), NoLLM)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    assert isinstance(default_client(), AnthropicClient)


def test_cache_key_depends_on_prompt_version_model_and_input():
    other_version = LLMRequest("judge-v2", REQ.model, REQ.system, REQ.user, 512, REQ.schema)
    other_input = LLMRequest(REQ.prompt_version, REQ.model, REQ.system, "x", 512, REQ.schema)
    keys = {REQ.cache_key, other_version.cache_key, other_input.cache_key}
    assert len(keys) == 3
    assert (
        REQ.input_hash
        == LLMRequest(*[REQ.prompt_version, "m2", REQ.system, REQ.user, 9, REQ.schema]).input_hash
    )


def test_cached_llm_only_caches_valid(db):
    calls = []

    class Flaky:
        def complete(self, request):
            calls.append(request)
            return "nope" if len(calls) == 1 else '{"v": 1}'

    llm = CachedLLM(Flaky(), db)
    assert llm.complete(REQ, parse_json_object) == {"v": 1}
    assert llm.complete(REQ, parse_json_object) == {"v": 1}
    assert len(calls) == 2
    assert db.execute("SELECT count(*) FROM llm_cache").fetchone()[0] == 1


def test_shipped_prompts_have_versions():
    assert load_prompt("judge").version == "judge-v1"
    assert load_prompt("tag").version == "tag-v1"


def test_parse_json_object_errors():
    with pytest.raises(MalformedResponse):
        parse_json_object("[1, 2]")
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
