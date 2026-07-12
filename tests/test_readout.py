"""Readout math: unembed vs llama.cpp, topk/ranks, lens vectors, edits."""

import numpy as np
import pytest

from jlens_gguf.lens import JacobianLensGGUF
from jlens_gguf.readout import LensReadout, compute_grid


def test_final_layer_readout_equals_model_logits(client, weights, readout):
    toks = client.tokenize("The little dog ran to the park and saw a")
    fr = client.forward(toks, dtype="f32", logits_positions=[3, len(toks) - 1])
    logits = readout.lens_logits(fr.activations[4], 4)
    for pos, ref in fr.logits.items():
        np.testing.assert_allclose(logits[pos], ref, atol=1e-4)
        assert logits[pos].argmax() == ref.argmax()


def test_topk_matches_argsort(rng):
    logits = rng.standard_normal((7, 100)).astype(np.float32)
    ids, vals = LensReadout.topk(logits, 5)
    ref = np.argsort(-logits, axis=-1)[:, :5]
    np.testing.assert_array_equal(ids, ref)
    np.testing.assert_allclose(vals, np.take_along_axis(logits, ref, axis=-1))


def test_ranks_of_matches_bruteforce(rng):
    logits = rng.standard_normal((9, 200)).astype(np.float32)
    targets = np.array([0, 7, 199, 42])
    ranks = LensReadout.ranks_of(logits, targets, chunk=4)
    for t in range(9):
        order = np.argsort(-logits[t])
        for j, tid in enumerate(targets):
            assert ranks[t, j] == list(order).index(tid)


def test_lens_vector_raises_token_logit(client, readout, weights):
    """Adding v_t to h must raise token t's lens logit more than others'."""
    toks = client.tokenize("Once upon a time there was a little")
    fr = client.forward(toks, dtype="f32")
    h = fr.activations[2][-1]
    tid = 300
    v = readout.lens_vector(2, tid, unit=True)
    scale = float(np.linalg.norm(h))
    before = readout.lens_logits(h, 2)
    after = readout.lens_logits(h + 2.0 * scale * v, 2)
    gain = after - before
    assert gain[tid] > 0
    assert (gain[tid] >= gain).mean() > 0.99  # among the very top gainers


def test_swap_factors_exchange_coordinates(client, readout):
    toks = client.tokenize("Once upon a time there was a little")
    fr = client.forward(toks, dtype="f32")
    h = fr.activations[2][-1]
    ta, tb = 300, 400
    va = readout.lens_vector(2, ta, unit=False)
    vb = readout.lens_vector(2, tb, unit=False)
    V = np.stack([va, vb], axis=1)
    c_before = np.linalg.pinv(V) @ h
    A, B = readout.swap_factors(2, ta, tb)
    h2 = h + A @ (B @ h)
    c_after = np.linalg.pinv(V) @ h2
    np.testing.assert_allclose(c_after, c_before[::-1], atol=1e-4)
    # orthogonal component untouched
    P = V @ np.linalg.pinv(V)
    np.testing.assert_allclose(h2 - P @ h2, h - P @ h, atol=1e-4)


def test_ablate_factors_remove_projection(readout, rng):
    h = rng.standard_normal(64).astype(np.float32) * 3
    tid = 123
    A, B = readout.ablate_factors(2, [tid])
    h2 = h + A @ (B @ h)
    v = readout.lens_vector(2, tid, unit=True)
    assert abs(float(v @ h2)) < 1e-4


def test_decompose_reduces_residual(client, readout):
    toks = client.tokenize("Once upon a time there was a little girl named")
    fr = client.forward(toks, dtype="f32")
    h = fr.activations[3][-1]
    items = readout.decompose(h, 3, k=8)
    assert 1 <= len(items) <= 8
    explained = [it["explained"] for it in items]
    assert all(b >= a - 1e-6 for a, b in zip(explained, explained[1:]))
    assert explained[-1] > 0.15  # a fitted layer should be decently explainable


def test_compute_grid(client, weights, readout):
    toks = client.tokenize("Once upon a time")
    fr = client.forward(toks, dtype="f32")
    layers = readout.readout_layers(weights.n_layers)
    grid = compute_grid(readout, fr.activations, layers, top_n=4)
    assert grid.top_ids.shape == (len(toks), len(layers), 4)
    # final-layer row equals model argmax
    final_logits = weights.unembed(fr.activations[4])
    np.testing.assert_array_equal(grid.top_ids[:, -1, 0], final_logits.argmax(-1))
    assert set(grid.norms) == set(layers)


def test_identity_lens_is_logit_lens(client, weights):
    toks = client.tokenize("Once upon a time")
    fr = client.forward(toks, dtype="f32")
    ident = JacobianLensGGUF.identity(d_model=64, layers=[2])
    ro = LensReadout(weights, ident)
    with_lens = ro.lens_logits(fr.activations[2], 2, use_lens=True)
    without = ro.lens_logits(fr.activations[2], 2, use_lens=False)
    np.testing.assert_allclose(with_lens, without, atol=1e-5)


def test_d_model_mismatch_raises(weights):
    bad = JacobianLensGGUF.identity(d_model=32, layers=[0])
    with pytest.raises(ValueError, match="d_model"):
        LensReadout(weights, bad)
