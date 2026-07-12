# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
"""GGUF-native lens fitting.

llama.cpp has no autograd, so the paper's causal estimator
``J_l = E[dh_final/dh_l]`` cannot be computed against a GGUF model directly.
This module fits the practical GGUF-native surrogate: per-layer ridge
regression of the final-layer residual on the layer-l residual, over a text
corpus,

    A_l = argmin_A  sum_p ||A h_l[p] (+ b) - h_final[p]||^2 + lambda ||A||^2

which is the same-position, correlational analogue of the Jacobian transport
(a tuned-lens-style affine translator, without the distribution-matching
objective). It only needs forward passes, streams two Gram matrices per
layer, and works on any quantized GGUF. For the paper's exact causal lens,
fit with the reference PyTorch code on the original checkpoint and convert
with :mod:`jlens_gguf.pt_convert`.

Like the reference, the first ``skip_first`` positions (attention sinks) and
the final position are excluded from the average.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence

import numpy as np

from jlens_gguf.client import NativeClient
from jlens_gguf.lens import JacobianLensGGUF

logger = logging.getLogger(__name__)

SKIP_FIRST_N_POSITIONS = 16


def fit_regression(
    client: NativeClient,
    prompts: Sequence[str],
    *,
    source_layers: Sequence[int] | None = None,
    target_layer: int | None = None,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
    max_seq_len: int = 128,
    ridge: float = 1e-4,
    affine: bool = True,
    base_model: str = "",
    progress: bool = True,
) -> JacobianLensGGUF:
    """Fit a regression lens over ``prompts`` via a running jlens-server.

    Args:
        client: connected :class:`~jlens_gguf.client.NativeClient`.
        prompts: text corpus (~100 prompts of ~128 tokens is usable; more is
            better, quality saturates quickly).
        source_layers: layers to fit (default: every layer below target).
        target_layer: transport target (default: final layer).
        skip_first: leading positions excluded from the average.
        max_seq_len: truncate each prompt to this many tokens.
        ridge: ridge strength, relative to ``trace(Sxx)/d``.
        affine: also fit a bias (recommended for regression lenses).
        base_model: informational tag stored in the lens file.
    """
    props = client.props()
    n_layers, d = props["n_layer"], props["n_embd"]
    target = (n_layers - 1) if target_layer is None else target_layer % n_layers
    layers = list(source_layers) if source_layers is not None else list(range(target))
    layers = sorted({l % n_layers for l in layers})
    if any(l >= target for l in layers):
        raise ValueError(f"source layers must be < target layer {target}")

    sxx = {l: np.zeros((d, d), dtype=np.float64) for l in layers}
    sxy = {l: np.zeros((d, d), dtype=np.float64) for l in layers}
    sx = {l: np.zeros(d, dtype=np.float64) for l in layers}
    sy = np.zeros(d, dtype=np.float64)
    h_ss = {l: 0.0 for l in layers}  # for h_rms
    n_samples = 0
    n_done = 0

    capture = sorted(set(layers) | {target})
    for i, prompt in enumerate(prompts):
        t0 = time.time()
        tokens = client.tokenize(prompt)[:max_seq_len]
        if len(tokens) <= skip_first + 1:
            logger.warning("skipping prompt %d: too short (%d tokens)", i, len(tokens))
            continue
        fr = client.forward(tokens, capture_layers=capture, dtype="f32")
        valid = slice(skip_first, len(tokens) - 1)
        Y = fr.activations[target][valid].astype(np.float64)
        for l in layers:
            X = fr.activations[l][valid].astype(np.float64)
            sxx[l] += X.T @ X
            sxy[l] += X.T @ Y
            sx[l] += X.sum(axis=0)
            h_ss[l] += float((X * X).sum())
        sy += Y.sum(axis=0)
        n_samples += Y.shape[0]
        n_done += 1
        if progress:
            logger.info(
                "prompt %d/%d  n_pos=%d  total=%d  %.1fs",
                i + 1, len(prompts), Y.shape[0], n_samples, time.time() - t0,
            )

    if n_samples == 0:
        raise ValueError("no prompts were long enough to fit on")

    jacobians: dict[int, np.ndarray] = {}
    biases: dict[int, np.ndarray] = {}
    h_rms: dict[int, float] = {}
    mu_y = sy / n_samples
    for l in layers:
        mu_x = sx[l] / n_samples
        if affine:
            cxx = sxx[l] - n_samples * np.outer(mu_x, mu_x)
            cxy = sxy[l] - n_samples * np.outer(mu_x, mu_y)
        else:
            cxx, cxy = sxx[l], sxy[l]
        lam = ridge * (np.trace(cxx) / d)
        At = np.linalg.solve(cxx + lam * np.eye(d), cxy)  # solves (X'X) A^T = X'Y
        A = At.T
        jacobians[l] = A.astype(np.float32)
        if affine:
            biases[l] = (mu_y - A @ mu_x).astype(np.float32)
        h_rms[l] = float(np.sqrt(h_ss[l] / (n_samples * d)) * np.sqrt(d))  # mean ||h||

    return JacobianLensGGUF(
        jacobians,
        d_model=d,
        n_prompts=n_done,
        target_layer=target,
        fit_method="regression",
        base_model=base_model or props.get("model_path", ""),
        biases=biases if affine else None,
        h_rms=h_rms,
    )


def load_corpus(path_or_spec: str, *, n_prompts: int = 100, min_chars: int = 400) -> list[str]:
    """Load a fitting corpus.

    - ``wikitext[:N]``: stream WikiText-103 rows from the HuggingFace
      datasets-server API (no `datasets` package needed).
    - anything else: a local text file; blank-line-separated blocks of at
      least ``min_chars`` characters become prompts.
    """
    if path_or_spec.startswith("wikitext"):
        if ":" in path_or_spec:
            n_prompts = int(path_or_spec.split(":", 1)[1])
        return _wikitext_prompts(n_prompts, min_chars=min_chars)
    with open(path_or_spec, encoding="utf-8") as f:
        text = f.read()
    blocks = [b.strip() for b in text.split("\n\n")]
    prompts = [b for b in blocks if len(b) >= min_chars]
    if not prompts:
        # fall back to fixed-size character windows
        prompts = [text[i : i + 2000] for i in range(0, len(text), 2000)]
        prompts = [p for p in prompts if len(p) >= min_chars]
    return prompts[:n_prompts]


def _wikitext_prompts(n_prompts: int, *, min_chars: int = 400) -> list[str]:
    import requests

    prompts: list[str] = []
    offset = 0
    while len(prompts) < n_prompts and offset < 50000:
        r = requests.get(
            "https://datasets-server.huggingface.co/rows",
            params={
                "dataset": "Salesforce/wikitext",
                "config": "wikitext-103-raw-v1",
                "split": "train",
                "offset": offset,
                "length": 100,
            },
            timeout=60,
        )
        r.raise_for_status()
        rows = r.json().get("rows", [])
        if not rows:
            break
        for row in rows:
            text = row["row"]["text"]
            if len(text.strip()) >= min_chars:
                prompts.append(text)
                if len(prompts) == n_prompts:
                    break
        offset += 100
    if len(prompts) < n_prompts:
        logger.warning("only found %d/%d wikitext prompts", len(prompts), n_prompts)
    return prompts
