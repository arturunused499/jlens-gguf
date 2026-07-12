"""Lens GGUF container: save/load roundtrip, identity, merge, .pt conversion."""

import pickle
import sys
import types
import zipfile

import numpy as np
import pytest

from jlens_gguf.lens import JacobianLensGGUF
from jlens_gguf.pt_convert import convert_pt_lens, load_pt


def test_roundtrip(tmp_path, rng):
    J = {2: rng.standard_normal((16, 16)).astype(np.float32),
         5: rng.standard_normal((16, 16)).astype(np.float32)}
    b = {2: rng.standard_normal(16).astype(np.float32)}
    lens = JacobianLensGGUF(
        J, d_model=16, n_prompts=42, target_layer=7, fit_method="regression",
        base_model="m.gguf", biases=b, h_rms={2: 1.5, 5: 2.5},
    )
    path = tmp_path / "lens.gguf"
    lens.save(str(path), dtype=np.float32)

    back = JacobianLensGGUF.load(str(path))
    assert back.source_layers == [2, 5]
    assert back.d_model == 16 and back.n_prompts == 42 and back.target_layer == 7
    assert back.fit_method == "regression" and back.base_model == "m.gguf"
    np.testing.assert_allclose(back.jacobians[2], J[2], rtol=0, atol=0)
    np.testing.assert_allclose(back.biases[2], b[2], rtol=0, atol=0)
    assert back.h_rms == {2: 1.5, 5: 2.5}


def test_roundtrip_f16_quantizes_but_close(tmp_path, rng):
    J = {0: (rng.standard_normal((8, 8)) * 0.05).astype(np.float32)}
    lens = JacobianLensGGUF(J, d_model=8)
    path = tmp_path / "lens16.gguf"
    lens.save(str(path), dtype=np.float16)
    back = JacobianLensGGUF.load(str(path))
    np.testing.assert_allclose(back.jacobians[0], J[0], atol=1e-4)


def test_transport_and_bias(rng):
    J = rng.standard_normal((8, 8)).astype(np.float32)
    b = rng.standard_normal(8).astype(np.float32)
    lens = JacobianLensGGUF({0: J}, d_model=8, biases={0: b})
    h = rng.standard_normal((3, 8)).astype(np.float32)
    np.testing.assert_allclose(lens.transport(h, 0), h @ J.T + b, rtol=1e-6)


def test_identity_lens():
    lens = JacobianLensGGUF.identity(d_model=4, layers=[0, 1])
    h = np.arange(4, dtype=np.float32)
    np.testing.assert_array_equal(lens.transport(h, 1), h)
    assert lens.fit_method == "identity"


def test_merge(rng):
    J1 = {0: np.full((4, 4), 1.0, np.float32)}
    J2 = {0: np.full((4, 4), 4.0, np.float32)}
    a = JacobianLensGGUF(J1, d_model=4, n_prompts=1)
    b = JacobianLensGGUF(J2, d_model=4, n_prompts=2)
    merged = JacobianLensGGUF.merge([a, b])
    np.testing.assert_allclose(merged.jacobians[0], np.full((4, 4), 3.0))
    assert merged.n_prompts == 3


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        JacobianLensGGUF({0: np.zeros((3, 4), np.float32)}, d_model=4)


# --------------------------------------------------------------------- #
# .pt conversion without torch: craft a torch-format zip by hand
# --------------------------------------------------------------------- #


def _fake_torch_save(path, d_model=6, layers=(1, 3), dtype=np.float16):
    """Write a torch.save-compatible zip (pickle + raw storages) without torch."""
    storages = {}

    class StubStorageRef:
        def __init__(self, key):
            self.key = key

    # fake torch modules so pickle emits GLOBAL torch._utils _rebuild_tensor_v2
    torch_mod = types.ModuleType("torch")
    utils_mod = types.ModuleType("torch._utils")

    def _rebuild_tensor_v2(storage, offset, size, stride, requires_grad, hooks, metadata=None):
        raise RuntimeError("never called at save time")

    utils_mod._rebuild_tensor_v2 = _rebuild_tensor_v2
    _rebuild_tensor_v2.__module__ = "torch._utils"
    _rebuild_tensor_v2.__qualname__ = "_rebuild_tensor_v2"
    torch_mod._utils = utils_mod
    added = [m for m in ("torch", "torch._utils") if m not in sys.modules]
    sys.modules.setdefault("torch", torch_mod)
    sys.modules.setdefault("torch._utils", utils_mod)

    class FakeTensor:
        def __init__(self, arr, key):
            self.arr = arr
            self.key = key

        def __reduce_ex__(self, protocol):
            size = self.arr.shape
            stride = tuple(s // self.arr.itemsize for s in self.arr.strides)
            return (
                utils_mod._rebuild_tensor_v2,
                (StubStorageRef(self.key), 0, size, stride, False, {}),
            )

    class Pickler(pickle.Pickler):
        def persistent_id(self, obj):
            if isinstance(obj, StubStorageRef):
                arr = storages[obj.key]
                storage_type = {np.float16: "HalfStorage", np.float32: "FloatStorage"}[arr.dtype.type]
                return ("storage", storage_type, obj.key, "cpu", arr.size)
            return None

    rng = np.random.default_rng(1)
    J = {}
    tensors = {}
    for i, l in enumerate(layers):
        arr = (rng.standard_normal((d_model, d_model)) * 0.1).astype(dtype)
        storages[str(i)] = arr
        tensors[l] = arr
        J[l] = FakeTensor(arr, str(i))

    obj = {"J": J, "n_prompts": 9, "source_layers": list(layers), "d_model": d_model}

    import io

    buf = io.BytesIO()
    try:
        Pickler(buf, protocol=2).dump(obj)
    finally:
        # remove the fake modules so pt_convert's `import torch` fails cleanly
        # and the pure-python loader path is exercised
        for m in added:
            sys.modules.pop(m, None)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("archive/data.pkl", buf.getvalue())
        for key, arr in storages.items():
            zf.writestr(f"archive/data/{key}", arr.tobytes())
    return {l: a.astype(np.float32) for l, a in tensors.items()}


def test_load_pt_pure_python(tmp_path):
    path = tmp_path / "lens.pt"
    expected = _fake_torch_save(str(path))
    ckpt = load_pt(str(path))
    assert ckpt["n_prompts"] == 9 and ckpt["d_model"] == 6
    for l, arr in expected.items():
        np.testing.assert_allclose(ckpt["J"][l].astype(np.float32), arr, atol=1e-3)


def test_convert_pt_lens(tmp_path):
    pt = tmp_path / "lens.pt"
    out = tmp_path / "lens.gguf"
    expected = _fake_torch_save(str(pt))
    lens = convert_pt_lens(str(pt), str(out), base_model="orig-model")
    assert lens.fit_method == "jacobian"
    back = JacobianLensGGUF.load(str(out))
    assert back.source_layers == [1, 3]
    for l, arr in expected.items():
        np.testing.assert_allclose(back.jacobians[l], arr, atol=1e-3)
