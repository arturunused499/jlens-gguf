"""Mixture-of-Experts capability check.

Skipped unless a MoE GGUF is available (it's large, so not in CI/the repo):
set JLENS_MOE_MODEL=/path/to/moe.gguf, or drop an OLMoE GGUF at
models/olmoe-1b-7b.gguf. Proves capture + numpy readout + a fit + steering all
work on a real MoE (routing/expert count don't touch the residual stream, so
the lens is architecture-agnostic).
"""

import os
import socket
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
MOE_MODEL = os.environ.get("JLENS_MOE_MODEL") or str(ROOT / "models" / "olmoe-1b-7b.gguf")
NATIVE_BIN = ROOT / "native" / "jlens-server"
PORT = int(os.environ.get("JLENS_MOE_PORT", "18198"))

pytestmark = pytest.mark.skipif(
    not Path(MOE_MODEL).exists() or not NATIVE_BIN.exists(),
    reason=f"no MoE model at {MOE_MODEL} (set JLENS_MOE_MODEL) or native server not built",
)


def _port_open(port):
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


@pytest.fixture(scope="module")
def moe_server():
    proc = None
    if not _port_open(PORT):
        proc = subprocess.Popen(
            [str(NATIVE_BIN), "-m", MOE_MODEL, "--port", str(PORT), "-c", "1024", "-b", "256"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        for _ in range(240):
            if _port_open(PORT):
                break
            time.sleep(0.5)
        else:
            proc.terminate()
            pytest.fail("MoE server did not start")
    yield f"http://127.0.0.1:{PORT}"
    if proc is not None:
        proc.terminate()
        proc.wait(timeout=10)


def test_moe_capture_and_readout(moe_server):
    from jlens_gguf.client import NativeClient
    from jlens_gguf.model_reader import ReadoutWeights

    client = NativeClient(moe_server)
    props = client.props()
    assert props["l_out_ok"] is True, "MoE residual capture must work"

    weights = ReadoutWeights.from_gguf(MOE_MODEL)
    toks = client.tokenize("The capital of France is")
    fr = client.forward(toks, dtype="f32", logits_positions=[-1])
    # every layer captured at full width
    assert sorted(fr.activations) == list(range(props["n_layer"]))
    for act in fr.activations.values():
        assert act.shape == (len(toks), props["n_embd"])
    # numpy readout of the final residual matches llama.cpp's own logits
    pos = len(toks) - 1
    mine = weights.unembed(fr.activations[props["n_layer"] - 1][pos])
    corr = float(np.corrcoef(mine, fr.logits[pos])[0, 1])
    assert corr > 0.999
    assert mine.argmax() == fr.logits[pos].argmax()


def test_moe_intervention_and_generation(moe_server, rng):
    from jlens_gguf.client import NativeClient

    client = NativeClient(moe_server)
    toks = client.tokenize("Paris is the capital of the country of")
    base = client.forward(toks, dtype="f32")
    # a mid-layer edit propagates through the MoE FFN to the final logits
    vec = (rng.standard_normal(base.activations[0].shape[1]) * 5).astype(np.float32)
    fr = client.forward(toks, dtype="f32", interventions=[
        {"layer": 4, "pos_start": 0, "pos_end": -1, "mode": "add", "vector": vec},
    ])
    np.testing.assert_array_equal(fr.activations[3], base.activations[3])  # earlier layer untouched
    assert np.abs(fr.activations[5] - base.activations[5]).max() > 1e-3     # later layer changed

    # steered generation differs from baseline
    g0 = client.forward(toks, capture=False, n_predict=6, sampling={"greedy": True})
    g1 = client.forward(toks, capture=False, n_predict=6, sampling={"greedy": True},
                        interventions=[{"layer": 4, "pos_start": 0, "pos_end": -1,
                                        "mode": "add", "vector": vec}])
    assert [g["token"] for g in g0.generated] != [g["token"] for g in g1.generated]
