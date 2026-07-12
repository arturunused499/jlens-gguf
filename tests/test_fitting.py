"""Regression fitting: transport quality and metadata."""

import numpy as np

from jlens_gguf.lens import JacobianLensGGUF
from jlens_gguf.readout import LensReadout


def test_fit_metadata(fitted_lens):
    assert fitted_lens.fit_method == "regression"
    assert fitted_lens.source_layers == [0, 1, 2, 3]
    assert fitted_lens.target_layer == 4
    assert fitted_lens.n_prompts == 12
    assert set(fitted_lens.biases) == {0, 1, 2, 3}
    assert all(v > 0 for v in fitted_lens.h_rms.values())


def test_fitted_beats_identity(client, weights, fitted_lens):
    """The fitted lens must agree with the model's output far more often than
    the raw logit lens on held-out text."""
    ident = JacobianLensGGUF.identity(d_model=64, layers=list(range(4)))
    ro_fit = LensReadout(weights, fitted_lens)
    ro_id = LensReadout(weights, ident)
    toks = client.tokenize(
        "The bird flew over the house and landed on the fence. A boy looked up"
    )
    fr = client.forward(toks, dtype="f32")
    model_top1 = weights.unembed(fr.activations[4]).argmax(-1)
    for layer in (1, 2, 3):
        fit_top1 = ro_fit.lens_logits(fr.activations[layer], layer).argmax(-1)
        id_top1 = ro_id.lens_logits(fr.activations[layer], layer).argmax(-1)
        fit_acc = float((fit_top1 == model_top1).mean())
        id_acc = float((id_top1 == model_top1).mean())
        assert fit_acc >= id_acc, f"layer {layer}: fitted {fit_acc} < identity {id_acc}"
    # and clearly better at the mid layer
    fit2 = float((ro_fit.lens_logits(fr.activations[2], 2).argmax(-1) == model_top1).mean())
    id2 = float((ro_id.lens_logits(fr.activations[2], 2).argmax(-1) == model_top1).mean())
    assert fit2 > id2


def test_fit_transport_mse(client, fitted_lens):
    """Transported mid-layer residuals should approximate final residuals
    better than untransported ones."""
    # held-out but in-distribution (the fixture corpus is TinyStories-style;
    # a 4-prompt fit on a 260K-param model does not transfer out of domain)
    toks = client.tokenize(
        "The little dog ran fast to the park and saw a big yellow ball near the pond."
    )
    fr = client.forward(toks, dtype="f32")
    valid = slice(4, len(toks) - 1)  # matches the fit's skip_first / last-position mask
    h_final = fr.activations[4][valid]
    for layer in (1, 2):
        h = fr.activations[layer][valid]
        err_lens = np.linalg.norm(fitted_lens.transport(h, layer) - h_final)
        err_raw = np.linalg.norm(h - h_final)
        assert err_lens < err_raw
