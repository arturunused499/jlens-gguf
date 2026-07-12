import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODEL = ROOT / "models" / "stories260K.gguf"
NATIVE_BIN = ROOT / "native" / "jlens-server"
NATIVE_PORT = int(os.environ.get("JLENS_TEST_PORT", "18191"))
NATIVE_URL = f"http://127.0.0.1:{NATIVE_PORT}"


def _port_open(port):
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


@pytest.fixture(scope="session")
def native_server():
    """A jlens-server running on the tiny test model."""
    if not MODEL.exists():
        pytest.skip(f"test model missing: {MODEL}")
    if not NATIVE_BIN.exists():
        pytest.skip(f"native server not built: {NATIVE_BIN} (run native/build.sh)")
    proc = None
    if not _port_open(NATIVE_PORT):
        proc = subprocess.Popen(
            [str(NATIVE_BIN), "-m", str(MODEL), "--port", str(NATIVE_PORT),
             "-c", "512", "--chunk", "128"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        for _ in range(120):
            if _port_open(NATIVE_PORT):
                break
            time.sleep(0.25)
        else:
            proc.terminate()
            pytest.fail("native server did not start")
    yield NATIVE_URL
    if proc is not None:
        proc.terminate()
        proc.wait(timeout=10)


@pytest.fixture(scope="session")
def client(native_server):
    from jlens_gguf.client import NativeClient

    return NativeClient(native_server)


@pytest.fixture(scope="session")
def weights():
    from jlens_gguf.model_reader import ReadoutWeights

    return ReadoutWeights.from_gguf(str(MODEL))


@pytest.fixture(scope="session")
def fitted_lens(client):
    """A small regression lens fitted on story-like prompts."""
    from jlens_gguf.fitting import fit_regression

    prompts = [
        "Once upon a time there was a little girl named Lily. She liked to play in "
        "the garden with her dog. One day she found a shiny red ball under the tree.",
        "Tom was a small boy who loved trucks. Every morning he would run to the "
        "window to watch the big red truck drive down the road. His mom made toast.",
        "The sun was warm and the birds sang. Anna and her brother walked to the "
        "park. They saw a duck in the pond and threw bread to it. The duck quacked.",
        "One day a cat named Max climbed a tall tree. He could not get down and he "
        "was scared. A kind man brought a ladder and helped him. Everyone was glad.",
    ] * 3
    return fit_regression(client, prompts, skip_first=4, max_seq_len=96, progress=False)


@pytest.fixture(scope="session")
def readout(weights, fitted_lens):
    from jlens_gguf.readout import LensReadout

    return LensReadout(weights, fitted_lens)


@pytest.fixture(scope="session")
def app(native_server, fitted_lens, tmp_path_factory):
    from jlens_gguf.server import App

    lens_path = tmp_path_factory.mktemp("lens") / "lens.gguf"
    fitted_lens.save(str(lens_path))
    return App(model_path=str(MODEL), native_url=native_server, lens_path=str(lens_path))


@pytest.fixture()
def rng():
    return np.random.default_rng(0)
