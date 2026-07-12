"""Backend mode: OpenAI-compatible completions with the live intervention set."""

import json

import numpy as np
import pytest
import requests


@pytest.fixture(autouse=True)
def clean_live_set(client):
    client.live_interventions_clear()
    yield
    client.live_interventions_clear()


CHAT_BODY = {
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "temperature": 0,
    "max_tokens": 12,
}


def _chat(native_server, body=None):
    r = requests.post(f"{native_server}/v1/chat/completions", json=body or CHAT_BODY, timeout=120)
    assert r.status_code == 200, r.text
    return r.json()


def test_models_endpoint(native_server):
    out = requests.get(f"{native_server}/v1/models", timeout=30).json()
    assert out["object"] == "list" and len(out["data"]) == 1


def test_chat_completion_shape_and_determinism(native_server):
    a = _chat(native_server)
    b = _chat(native_server)
    choice = a["choices"][0]
    assert choice["message"]["role"] == "assistant"
    assert isinstance(choice["message"]["content"], str) and choice["message"]["content"]
    assert choice["finish_reason"] in ("stop", "length")
    usage = a["usage"]
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
    assert a["choices"][0]["message"]["content"] == b["choices"][0]["message"]["content"]


def test_text_completion(native_server):
    r = requests.post(f"{native_server}/v1/completions", json={
        "prompt": "Once upon a time", "temperature": 0, "max_tokens": 8}, timeout=120)
    out = r.json()
    assert out["object"] == "text_completion"
    assert out["choices"][0]["text"]


def test_streaming_matches_nonstream(native_server):
    full = _chat(native_server)["choices"][0]["message"]["content"]
    r = requests.post(f"{native_server}/v1/chat/completions",
                      json={**CHAT_BODY, "stream": True}, stream=True, timeout=120)
    assert r.headers["content-type"].startswith("text/event-stream")
    pieces, finish, saw_done = [], None, False
    for line in r.iter_lines():
        if not line or not line.startswith(b"data: "):
            continue
        payload = line[6:]
        if payload == b"[DONE]":
            saw_done = True
            break
        chunk = json.loads(payload)
        delta = chunk["choices"][0]["delta"]
        if "content" in delta:
            pieces.append(delta["content"])
        if chunk["choices"][0]["finish_reason"]:
            finish = chunk["choices"][0]["finish_reason"]
    assert saw_done and finish in ("stop", "length")
    assert "".join(pieces) == full


def test_stop_sequence(native_server):
    out = requests.post(f"{native_server}/v1/completions", json={
        "prompt": "Once upon a time, there was a little girl named",
        "temperature": 0, "max_tokens": 30, "stop": ["."]}, timeout=120).json()
    choice = out["choices"][0]
    assert "." not in choice["text"]
    assert choice["finish_reason"] == "stop"


def test_empty_stop_string_is_ignored(native_server):
    """An empty stop string must not terminate every completion immediately."""
    out = requests.post(f"{native_server}/v1/completions", json={
        "prompt": "Once upon a time", "temperature": 0, "max_tokens": 8,
        "stop": [""]}, timeout=120).json()
    choice = out["choices"][0]
    assert out["usage"]["completion_tokens"] == 8
    assert choice["finish_reason"] == "length"
    assert choice["text"]


def test_live_interventions_affect_completions(native_server, client, rng):
    base = _chat(native_server)["choices"][0]["message"]["content"]

    vec = (rng.standard_normal(64) * 4).astype(np.float32)
    out = client.live_interventions_set(
        [{"layer": 2, "pos_start": 0, "pos_end": -1, "mode": "add", "vector": vec}],
        meta={"ui_specs": [{"type": "steer"}]},
    )
    assert out["count"] == 1
    state = client.live_interventions_get()
    assert state["count"] == 1
    assert state["meta"]["ui_specs"] == [{"type": "steer"}]
    assert "data" not in state["interventions"][0]  # bulky payload not echoed

    steered = _chat(native_server)["choices"][0]["message"]["content"]
    assert steered != base

    client.live_interventions_clear()
    restored = _chat(native_server)["choices"][0]["message"]["content"]
    assert restored == base


def test_prefix_cache_reuses_and_is_deterministic(native_server):
    turn1 = _chat(native_server)
    conv = CHAT_BODY["messages"] + [
        {"role": "assistant", "content": turn1["choices"][0]["message"]["content"]},
        {"role": "user", "content": "Continue"},
    ]
    body2 = {"messages": conv, "temperature": 0, "max_tokens": 10}
    warm = _chat(native_server, body2)
    assert warm["timings"]["cached_tokens"] > 0  # reused turn-1 KV
    assert warm["choices"][0]["message"]["content"]  # a valid continuation
    # deterministic from the same (warm) cache state
    warm2 = _chat(native_server, body2)
    assert warm2["choices"][0]["message"]["content"] == warm["choices"][0]["message"]["content"]
    # NOTE: warm vs a fully-fresh recompute may differ by a token — llama.cpp's
    # attention sums cached and in-batch KV blocks in a different order, so
    # greedy output is not bit-invariant to prefix reuse (same as llama-server).
    # The visualizer's forward path always recomputes fresh, so readouts there
    # are exact and reproducible.


def test_last_completion_and_forward_interplay(native_server, client):
    out = _chat(native_server)
    last = client.last_completion()
    assert last["n_prompt"] + last["n_gen"] == len(last["tokens"])
    assert last["n_gen"] == out["usage"]["completion_tokens"]
    # a visualizer forward on those exact tokens works and clears the KV cache
    fr = client.forward(last["tokens"], capture_layers=[2], dtype="f16")
    assert fr.activations[2].shape[0] == len(last["tokens"])
    # completions still work afterwards (cache was invalidated, not corrupted)
    again = _chat(native_server)
    assert again["choices"][0]["message"]["content"]
    assert again["usage"]["completion_tokens"] > 0


def test_bad_requests(native_server):
    r = requests.post(f"{native_server}/v1/chat/completions", json={"foo": 1}, timeout=30)
    assert r.status_code == 400 and "message" in r.json()["error"]
    r = requests.post(f"{native_server}/v1/completions", json={"prompt": ["a", "b"]}, timeout=30)
    assert r.status_code == 400


def test_quickstart_model_discovery(native_server):
    """The quickstart path reads the served model from a jlens/llama server's
    /props (jlens-server's /props also carries model_path)."""
    from jlens_gguf.cli import _model_from_llama_server

    path = _model_from_llama_server(native_server)
    assert path is not None and path.endswith("stories260K.gguf")
    # unreachable server -> None, no exception
    assert _model_from_llama_server("http://127.0.0.1:59999") is None


def test_bridge_live_endpoints(app, native_server):
    tid = app.api_search_tokens("happ")["results"][0]["token"]
    res = app.api_live_push({"interventions": [
        {"type": "steer", "token_id": tid, "alpha": 6.0, "layers": [1, 3], "pos": [5, 9]},
    ]})
    assert res["count"] >= 1
    assert res["specs"][0]["pos"] == [0, -1]  # positions dropped for live use

    state = app.api_live_state()
    assert state["count"] >= 1
    assert state["openai_url"].endswith("/v1")
    assert state["ui_specs"][0]["token_id"] == tid

    _chat(native_server)  # serve one completion through the steered backend
    last = app.api_live_last()
    assert len(last["pieces"]) == len(last["tokens"])
    assert last["n_gen"] > 0

    app.api_live_clear({})
    assert app.api_live_state()["count"] == 0
