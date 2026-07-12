# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
"""jlens_gguf: the Jacobian lens for GGUF models served by llama.cpp.

A GGUF-native re-implementation of Anthropic's Jacobian-lens reference code
(https://transformer-circuits.pub/2026/workspace/index.html): the lens lives
in a GGUF file, the model weights needed for readout (final norm +
unembedding) are read from the model's GGUF, all math is numpy, and residual
activations plus live interventions come from ``jlens-server`` — a small
llama.cpp-based activation server that mmaps the same GGUF the user's
llama-server does.

Layout:

- :mod:`jlens_gguf.lens`         GGUF lens container (load/save/convert/identity)
- :mod:`jlens_gguf.model_reader` readout weights from a model GGUF
- :mod:`jlens_gguf.readout`      numpy lens math: transport, unembed, ranks,
                                 lens vectors, steering/patching, decomposition
- :mod:`jlens_gguf.client`       HTTP client for jlens-server
- :mod:`jlens_gguf.fitting`      GGUF-native lens fitting (ridge regression)
- :mod:`jlens_gguf.pt_convert`   pure-python converter for reference .pt lenses
- :mod:`jlens_gguf.server`       the bridge web server + interactive UI
"""

from jlens_gguf.lens import JacobianLensGGUF
from jlens_gguf.model_reader import ReadoutWeights
from jlens_gguf.readout import LensReadout
from jlens_gguf.client import NativeClient, ForwardResult

__version__ = "0.1.0"

__all__ = [
    "JacobianLensGGUF",
    "ReadoutWeights",
    "LensReadout",
    "NativeClient",
    "ForwardResult",
]
