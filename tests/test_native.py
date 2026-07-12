"""Native jlens-server: capture, determinism, interventions, generation."""

import numpy as np
import pytest


@pytest.fixture(scope="module")
def toks(client):
    return client.tokenize("Once upon a time there was a little girl named Lily. She liked to")


def test_props(client):
    props = client.props()
    assert props["l_out_ok"] is True
    assert props["n_layer"] == 5 and props["n_embd"] == 64 and props["n_vocab"] == 512


def test_tokenize_detokenize(client):
    toks, pieces = client.tokenize_with_pieces("Hello world")
    assert len(toks) == len(pieces) >= 2
    assert "Hello world" in client.detokenize(toks)


def test_vocab(client):
    vocab = client.vocab()
    assert len(vocab) == 512
    assert all(isinstance(p, str) for p in vocab)


def test_capture_shapes_and_determinism(client, toks):
    fr1 = client.forward(toks, dtype="f32")
    fr2 = client.forward(toks, dtype="f32")
    assert sorted(fr1.activations) == list(range(5))
    for l, act in fr1.activations.items():
        assert act.shape == (len(toks), 64)
        np.testing.assert_array_equal(act, fr2.activations[l])


def test_capture_subset_and_f16(client, toks):
    fr = client.forward(toks, capture_layers=[1, 3], dtype="f16")
    assert sorted(fr.activations) == [1, 3]
    fr32 = client.forward(toks, capture_layers=[1, 3], dtype="f32")
    np.testing.assert_allclose(fr.activations[1], fr32.activations[1], atol=2e-3, rtol=1e-2)


def test_logits_positions(client, toks, weights):
    fr = client.forward(toks, dtype="f32", logits_positions=[-1])
    pos = len(toks) - 1
    assert pos in fr.logits and fr.logits[pos].shape == (512,)
    # numpy unembed of the captured final layer must match llama.cpp logits
    mine = weights.unembed(fr.activations[4][pos])
    np.testing.assert_allclose(mine, fr.logits[pos], atol=1e-4)


def test_add_intervention(client, toks):
    base = client.forward(toks, dtype="f32")
    vec = np.zeros(64, dtype=np.float32)
    vec[:4] = 3.0
    fr = client.forward(toks, dtype="f32", interventions=[
        {"layer": 2, "pos_start": 5, "pos_end": 9, "mode": "add", "vector": vec},
    ])
    delta = fr.activations[2] - base.activations[2]
    np.testing.assert_allclose(delta[5:9, :4], 3.0, atol=1e-5)
    np.testing.assert_allclose(delta[:5], 0.0, atol=1e-6)
    # the edit lands after block 2 produced l_out-2, so its other columns are untouched
    np.testing.assert_allclose(delta[9:, :], 0.0, atol=1e-6)
    # earlier layers untouched, later layers affected at pos >= 5
    np.testing.assert_array_equal(fr.activations[1], base.activations[1])
    assert np.abs(fr.activations[3][5:] - base.activations[3][5:]).max() > 1e-3


def test_set_intervention(client, toks):
    vec = np.linspace(-1, 1, 64).astype(np.float32)
    fr = client.forward(toks, dtype="f32", interventions=[
        {"layer": 1, "pos_start": 3, "pos_end": 4, "mode": "set", "vector": vec},
    ])
    np.testing.assert_allclose(fr.activations[1][3], vec, atol=1e-6)


def test_lowrank_intervention(client, toks, rng):
    base = client.forward(toks, dtype="f32")
    # project out a random unit direction: h -= v v^T h
    v = rng.standard_normal(64).astype(np.float32)
    v /= np.linalg.norm(v)
    A = -v[:, None]
    B = v[None, :]
    fr = client.forward(toks, dtype="f32", interventions=[
        {"layer": 2, "pos_start": 0, "pos_end": -1, "mode": "lowrank", "a": A, "b": B},
    ])
    proj = fr.activations[2] @ v
    np.testing.assert_allclose(proj, 0.0, atol=1e-4)
    expected = base.activations[2] - np.outer(base.activations[2] @ v, v)
    np.testing.assert_allclose(fr.activations[2], expected, atol=1e-4)


def test_generation(client, toks):
    fr = client.forward(toks, n_predict=8, dtype="f32", sampling={"greedy": True})
    assert 0 < fr.n_gen <= 8
    assert len(fr.tokens) == fr.n_prompt + fr.n_gen
    assert fr.activations[0].shape[0] == len(fr.tokens)
    assert len(fr.generated_text) > 0
    # greedy generation is deterministic
    fr2 = client.forward(toks, n_predict=8, dtype="f32", sampling={"greedy": True})
    assert [g["token"] for g in fr.generated] == [g["token"] for g in fr2.generated]


def test_generation_logits_positions(client, toks, weights):
    """Logits can be requested for generated positions (same semantics as
    prompt positions: the next-token distribution after consuming pos)."""
    fr = client.forward(toks, n_predict=5, dtype="f32", logits_positions=[-1],
                        sampling={"greedy": True})
    last = fr.n_prompt + fr.n_gen - 1
    assert last in fr.logits
    mine = weights.unembed(fr.activations[4][last])
    np.testing.assert_allclose(mine, fr.logits[last], atol=1e-4)


def test_generation_steered_differs(client, toks, rng):
    vec = (rng.standard_normal(64) * 4.0).astype(np.float32)
    fr0 = client.forward(toks, n_predict=8, capture=False, sampling={"greedy": True})
    fr1 = client.forward(toks, n_predict=8, capture=False, sampling={"greedy": True},
                         interventions=[{"layer": 2, "pos_start": 0, "pos_end": -1,
                                         "mode": "add", "vector": vec}])
    assert [g["token"] for g in fr0.generated] != [g["token"] for g in fr1.generated]


def test_client_session_is_thread_local(client, toks):
    """Each thread gets its own requests.Session (the bridge calls this client
    concurrently from ThreadingHTTPServer workers)."""
    import threading

    sessions = {}
    errors = []

    def worker(i):
        try:
            sessions[i] = id(client._session)
            for _ in range(5):
                assert client.props()["l_out_ok"] is True
                client.tokenize("hello world")
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len(set(sessions.values())) == len(threads)  # distinct session per thread


def test_bad_requests(client, toks):
    with pytest.raises(RuntimeError, match="out of range"):
        client.forward([999999])
    with pytest.raises(RuntimeError, match="layer out of range"):
        client.forward(toks, interventions=[
            {"layer": 99, "pos_start": 0, "pos_end": -1, "mode": "add",
             "vector": np.zeros(64, np.float32)}])
    with pytest.raises(RuntimeError, match="d_model"):
        client.forward(toks, interventions=[
            {"layer": 1, "pos_start": 0, "pos_end": -1, "mode": "add",
             "vector": np.zeros(32, np.float32)}])
