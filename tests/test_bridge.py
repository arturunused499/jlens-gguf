"""Bridge App API: slice, ranks, readout, decompose, interventions, generate."""

import base64

import numpy as np
import pytest

PROMPT = "Once upon a time there was a little girl named Lily. She wanted to"


def _grid(res):
    T, L, K = len(res["tokens"]), len(res["layers"]), res["top_n"]
    return np.frombuffer(base64.b64decode(res["top_ids"]), dtype="<i4").reshape(T, L, K)


def _ranks(res):
    return np.frombuffer(base64.b64decode(res["ranks"]), dtype="<i4").reshape(res["shape"])


def test_props(app):
    props = app.api_props()
    assert props["n_layers"] == 5
    assert props["lens"]["method"] == "regression"
    assert props["lens"]["source_layers"] == [0, 1, 2, 3]


def test_slice_and_pieces(app):
    res = app.api_slice({"prompt": PROMPT})
    grid = _grid(res)
    assert grid.shape[1] == 5 and grid.shape[2] == res["top_n"]
    assert len(res["pieces"]) == len(res["tokens"])
    assert res["n_gen"] == 0
    assert set(map(int, res["norms"].keys())) == set(res["layers"])
    # all grid ids are valid tokens
    assert grid.min() >= 0 and grid.max() < 512


def test_slice_with_generation(app):
    res = app.api_slice({"prompt": "Once upon a time", "n_predict": 6})
    assert res["n_gen"] > 0
    assert len(res["tokens"]) == res["n_prompt"] + res["n_gen"]
    assert len(res["generated_text"]) > 0


def test_ranks_consistent_with_grid(app):
    res = app.api_slice({"prompt": PROMPT})
    grid = _grid(res)
    t, li = len(res["tokens"]) - 1, 2
    top_tok = int(grid[t, li, 0])
    ranks = _ranks(app.api_ranks({"ctx_id": res["ctx_id"], "token_ids": [top_tok]}))
    assert ranks[t, li, 0] == 0  # the cell's top-1 has rank 0 there


def test_readout(app):
    res = app.api_slice({"prompt": PROMPT})
    grid = _grid(res)
    t = len(res["tokens"]) - 1
    ro = app.api_readout({"ctx_id": res["ctx_id"], "pos": t, "layer": 3, "top_n": 8})
    assert len(ro["tokens"]) == 8
    assert ro["tokens"][0]["token"] == int(grid[t, 3, 0])
    probs = [x["prob"] for x in ro["tokens"]]
    assert probs == sorted(probs, reverse=True)


def test_decompose(app):
    res = app.api_slice({"prompt": PROMPT})
    dec = app.api_decompose({"ctx_id": res["ctx_id"], "pos": len(res["tokens"]) - 1, "layer": 3, "k": 5})
    assert 1 <= len(dec["items"]) <= 5
    assert all("piece" in it for it in dec["items"])


def test_search_tokens(app):
    out = app.api_search_tokens("play")["results"]
    assert any(r["piece"].strip() == "play" for r in out)
    exact = app.api_search_tokens("#337")["results"]
    assert exact[0]["token"] == 337


def test_steer_intervention_moves_rank(app):
    tid = app.api_search_tokens("happ")["results"][0]["token"]
    base = app.api_slice({"prompt": PROMPT})
    steered = app.api_slice({
        "prompt": PROMPT,
        "interventions": [{"type": "steer", "token_id": tid, "alpha": 6.0,
                           "layers": [1, 3], "pos": [0, -1]}],
    })
    T = len(base["tokens"])
    r0 = _ranks(app.api_ranks({"ctx_id": base["ctx_id"], "token_ids": [tid]}))
    r1 = _ranks(app.api_ranks({"ctx_id": steered["ctx_id"], "token_ids": [tid]}))
    assert r1[T - 1, -1, 0] < r0[T - 1, -1, 0]
    assert r1[T - 1, -1, 0] <= 3


def test_negative_steer_suppresses(app):
    """Negative alpha should push the token's rank down (ablation-ish)."""
    base = app.api_slice({"prompt": PROMPT})
    grid = _grid(base)
    T = len(base["tokens"])
    top_tok = int(grid[T - 1, -1, 0])  # model's top-1 at last position
    steered = app.api_slice({
        "prompt": PROMPT,
        "interventions": [{"type": "steer", "token_id": top_tok, "alpha": -6.0,
                           "layers": [0, 3], "pos": [T - 1, -1]}],
    })
    r0 = _ranks(app.api_ranks({"ctx_id": base["ctx_id"], "token_ids": [top_tok]}))
    r1 = _ranks(app.api_ranks({"ctx_id": steered["ctx_id"], "token_ids": [top_tok]}))
    assert r0[T - 1, -1, 0] == 0
    assert r1[T - 1, -1, 0] > 0


def test_swap_intervention(app):
    base = app.api_slice({"prompt": PROMPT})
    grid = _grid(base)
    T = len(base["tokens"])
    play = int(grid[T - 1, 3, 0])
    happ = app.api_search_tokens("happ")["results"][0]["token"]
    res = app.api_slice({
        "prompt": PROMPT,
        "interventions": [{"type": "swap", "token_a": play, "token_b": happ,
                           "layers": [3, 3], "pos": [T - 1, -1]}],
    })
    ro = app.api_readout({"ctx_id": res["ctx_id"], "pos": T - 1, "layer": 3, "top_n": 3})
    assert ro["tokens"][0]["token"] == happ


def test_ablate_intervention(app):
    base = app.api_slice({"prompt": PROMPT})
    grid = _grid(base)
    T = len(base["tokens"])
    top3 = int(grid[T - 1, 3, 0])
    res = app.api_slice({
        "prompt": PROMPT,
        "interventions": [{"type": "ablate", "token_id": top3, "layers": [3, 3],
                           "pos": [T - 1, -1]}],
    })
    ro = app.api_readout({"ctx_id": res["ctx_id"], "pos": T - 1, "layer": 3, "top_n": 3})
    assert ro["tokens"][0]["token"] != top3


def test_generate_compare(app):
    tid = app.api_search_tokens("happ")["results"][0]["token"]
    out = app.api_generate({
        "prompt": "Once upon a time there was a",
        "n_predict": 10,
        "interventions": [{"type": "steer", "token_id": tid, "alpha": 6.0,
                           "layers": [1, 3], "pos": [0, -1]}],
        "compare": True,
    })
    assert out["steered"]["text"] != out["baseline"]["text"]


def test_use_lens_false_is_logit_lens(app):
    res = app.api_slice({"prompt": "Once upon a time", "use_lens": False})
    assert res["use_lens"] is False


def test_bad_ctx(app):
    with pytest.raises(ValueError, match="ctx_id"):
        app.api_ranks({"ctx_id": "nope", "token_ids": [1]})


def test_context_eviction(app):
    ids = [app.api_slice({"prompt": f"Once upon a time number {i}"})["ctx_id"] for i in range(4)]
    assert ids[0] not in app.contexts  # max_contexts = 3
    assert ids[-1] in app.contexts
