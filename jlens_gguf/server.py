# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
"""The jlens bridge: web UI + JSON API between the browser and jlens-server.

Serves the interactive visualizer (static files from ``jlens_gguf/web``) and
a JSON API. All lens math happens here in numpy; the native server only
produces residual activations and applies generic residual edits.

API (all JSON unless noted):
    GET  /api/props
    POST /api/tokenize     {content}
    POST /api/template     {messages: [{role, content}]}
    POST /api/slice        {prompt|tokens, use_lens?, top_n?, stride?,
                            n_predict?, sampling?, interventions?: [UI specs]}
    POST /api/ranks        {ctx_id, token_ids: [...]}
    POST /api/readout      {ctx_id, pos, layer, top_n?}
    POST /api/decompose    {ctx_id, pos, layer, k?}
    POST /api/generate     {prompt|tokens|ctx_id, n_predict, sampling?,
                            interventions?, compare?}
    GET  /api/search_tokens?q=...&limit=...

UI intervention specs (translated to native add/lowrank edits here):
    {"type": "steer",  "token_id": t, "alpha": a, "layers": [l0, l1]|null,
     "pos": [p0, p1]}                       # p1 = -1 -> open-ended
    {"type": "ablate", "token_id": t, "layers": ..., "pos": ...}
    {"type": "swap",   "token_a": s, "token_b": t, "layers": ..., "pos": ...}
"""

from __future__ import annotations

import base64
import json
import logging
import threading
import time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from jlens_gguf.client import NativeClient
from jlens_gguf.lens import JacobianLensGGUF
from jlens_gguf.model_reader import ReadoutWeights
from jlens_gguf.readout import LensReadout, compute_grid

logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).parent / "web"

MIME = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
}


def _b64(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode()


def safe_static_path(rel: str, root: Path = WEB_DIR) -> Path | None:
    """Resolve ``rel`` under ``root`` and return it only if it stays inside.

    Uses a path-boundary check (``root in target.parents``), not a string
    prefix, so ``../web-backup/x`` can't escape to a sibling directory whose
    name merely starts with the web dir's.
    """
    web_root = root.resolve()
    target = (web_root / rel.lstrip("/")).resolve()
    if target == web_root or web_root in target.parents:
        return target
    return None


class LogitsCache:
    """LRU cache of per-(ctx, layer) lens logits, bounded by bytes."""

    def __init__(self, budget_bytes: int = 1 << 30) -> None:
        self.budget = budget_bytes
        self._data: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self._bytes = 0

    def get(self, key: tuple) -> np.ndarray | None:
        arr = self._data.get(key)
        if arr is not None:
            self._data.move_to_end(key)
        return arr

    def put(self, key: tuple, arr: np.ndarray) -> None:
        if key in self._data:
            self._bytes -= self._data.pop(key).nbytes
        self._data[key] = arr
        self._bytes += arr.nbytes
        while self._bytes > self.budget and len(self._data) > 1:
            _, old = self._data.popitem(last=False)
            self._bytes -= old.nbytes

    def drop_ctx(self, ctx_id: str) -> None:
        for key in [k for k in self._data if k[0] == ctx_id]:
            self._bytes -= self._data.pop(key).nbytes


class Context:
    """One computed slice: tokens, activations, grid."""

    def __init__(self, ctx_id, tokens, n_prompt, n_gen, activations, grid, use_lens, interventions):
        self.ctx_id = ctx_id
        self.tokens = tokens
        self.n_prompt = n_prompt
        self.n_gen = n_gen
        self.activations = activations          # layer -> [T, d] f32
        self.grid = grid                        # SliceGrid
        self.use_lens = use_lens
        self.interventions = interventions      # UI specs used


class App:
    def __init__(
        self,
        *,
        model_path: str,
        native_url: str,
        lens_path: str | None = None,
        top_n: int = 10,
        max_contexts: int = 3,
        logits_cache_mb: int = 1024,
        max_tokens: int | None = None,
    ) -> None:
        self.client = NativeClient(native_url)
        self.props = self.client.props()
        self.n_layers = self.props["n_layer"]

        logger.info("reading readout weights from %s ...", model_path)
        self.weights = ReadoutWeights.from_gguf(model_path)
        if self.weights.n_layers != self.n_layers:
            raise ValueError("model GGUF disagrees with native server on layer count")

        if lens_path:
            self.lens = JacobianLensGGUF.load(lens_path)
            logger.info("loaded lens: %r", self.lens)
        else:
            self.lens = JacobianLensGGUF.identity(
                d_model=self.weights.d_model,
                layers=list(range(self.n_layers - 1)),
                base_model=model_path,
            )
            logger.info("no lens file given: using identity (logit lens) baseline")
        self.lens_path = lens_path
        self.readout = LensReadout(self.weights, self.lens)

        logger.info("fetching vocab from native server ...")
        self.vocab = self.client.vocab()
        self.vocab_attrs = self.client.vocab_attrs()
        self.vocab_lower = [p.lower() for p in self.vocab]

        self.top_n = top_n
        self.max_tokens = max_tokens or (self.props["n_ctx"] - 8)
        self.contexts: OrderedDict[str, Context] = OrderedDict()
        self.max_contexts = max_contexts
        self.logits_cache = LogitsCache(logits_cache_mb << 20)
        self.lock = threading.Lock()
        self._ctx_counter = 0

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #

    def pieces(self, tokens: list[int]) -> list[str]:
        return [self.vocab[t] if 0 <= t < len(self.vocab) else f"[{t}]" for t in tokens]

    def resolve_tokens(self, body: dict) -> list[int]:
        if body.get("tokens"):
            tokens = [int(t) for t in body["tokens"]]
        elif body.get("messages"):
            prompt = self.client.apply_template(body["messages"], add_assistant=True)
            if body.get("assistant_prefill"):
                prompt += body["assistant_prefill"]
            tokens = self.client.tokenize(prompt, add_special=True, parse_special=True)
        elif body.get("prompt") is not None:
            tokens = self.client.tokenize(
                body["prompt"],
                add_special=bool(body.get("add_special", True)),
                parse_special=bool(body.get("parse_special", True)),
            )
        else:
            raise ValueError("provide 'prompt', 'tokens', or 'messages'")
        if not tokens:
            raise ValueError("empty prompt")
        if len(tokens) > self.max_tokens:
            tokens = tokens[: self.max_tokens]
        return tokens

    def readout_layers(self, stride: int = 1) -> list[int]:
        layers = self.readout.readout_layers(self.n_layers)
        if stride > 1:
            final = layers[-1]
            layers = layers[::stride]
            if final not in layers:
                layers.append(final)
        return layers

    def layer_norms(self, tokens: list[int] | None, layers: list[int]) -> dict[int, float]:
        """Median residual norms for steering scale: lens h_rms if fitted,
        else from a cached context (matching the token prefix when given),
        else a dedicated capture pass."""
        norms = {l: self.lens.h_rms[l] for l in layers if l in self.lens.h_rms}
        missing = [l for l in layers if l not in norms]
        if not missing:
            return norms
        for ctx in reversed(self.contexts.values()):
            match = (
                tokens is None
                or ctx.tokens[: len(tokens)] == tokens
                or tokens[: len(ctx.tokens)] == ctx.tokens
            )
            if match and ctx.grid is not None:
                for l in list(missing):
                    if l in ctx.grid.norms:
                        norms[l] = ctx.grid.norms[l]
                        missing.remove(l)
                break
        if missing:
            probe = tokens or self.client.tokenize(
                "The quick brown fox jumps over the lazy dog."
            )
            fr = self.client.forward(probe, capture_layers=missing, dtype="f16")
            for l in missing:
                norms[l] = float(np.median(np.linalg.norm(fr.activations[l], axis=-1)))
        return norms

    def translate_interventions(self, specs: list[dict], tokens: list[int] | None) -> list[dict]:
        """UI intervention specs -> native add/lowrank edits."""
        if not specs:
            return []
        fitted = set(self.lens.source_layers)
        native: list[dict] = []

        def layer_range(spec) -> list[int]:
            lr = spec.get("layers")
            if lr is None:
                return sorted(fitted)
            l0, l1 = int(lr[0]), int(lr[1])
            return [l for l in sorted(fitted) if l0 <= l <= l1]

        steer_layers = sorted(
            {l for s in specs if s.get("type") == "steer" for l in layer_range(s)}
        )
        norms = self.layer_norms(tokens, steer_layers) if steer_layers else {}

        for spec in specs:
            kind = spec.get("type")
            pos = spec.get("pos") or [0, -1]
            p0, p1 = int(pos[0]), int(pos[1])
            if kind == "steer":
                tid = int(spec["token_id"])
                alpha = float(spec.get("alpha", 2.0))
                for l in layer_range(spec):
                    vec = self.readout.steer_vector(l, tid, alpha, h_rms=norms.get(l))
                    native.append({"layer": l, "pos_start": p0, "pos_end": p1, "mode": "add", "vector": vec})
            elif kind == "ablate":
                tid = int(spec["token_id"])
                for l in layer_range(spec):
                    A, B = self.readout.ablate_factors(l, [tid])
                    native.append({"layer": l, "pos_start": p0, "pos_end": p1, "mode": "lowrank", "a": A, "b": B})
            elif kind == "swap":
                ta, tb = int(spec["token_a"]), int(spec["token_b"])
                for l in layer_range(spec):
                    A, B = self.readout.swap_factors(l, ta, tb)
                    native.append({"layer": l, "pos_start": p0, "pos_end": p1, "mode": "lowrank", "a": A, "b": B})
            else:
                raise ValueError(f"unknown intervention type {kind!r}")
        return native

    def lens_logits_for(self, ctx: Context, layer: int) -> np.ndarray:
        key = (ctx.ctx_id, layer, ctx.use_lens)
        cached = self.logits_cache.get(key)
        if cached is not None:
            return cached
        logits = self.readout.lens_logits(ctx.activations[layer], layer, use_lens=ctx.use_lens)
        self.logits_cache.put(key, logits.astype(np.float32))
        return logits

    # ------------------------------------------------------------------ #
    # API handlers
    # ------------------------------------------------------------------ #

    def api_props(self) -> dict:
        return {
            "model": self.props,
            "model_name": self.weights.model_name or Path(self.props["model_path"]).name,
            "arch": self.weights.arch,
            "n_vocab": self.weights.n_vocab,
            "d_model": self.weights.d_model,
            "n_layers": self.n_layers,
            "lens": {
                "path": self.lens_path,
                "method": self.lens.fit_method,
                "n_prompts": self.lens.n_prompts,
                "source_layers": self.lens.source_layers,
                "target_layer": self.lens.target_layer,
                "has_bias": bool(self.lens.biases),
            },
            "top_n": self.top_n,
        }

    def api_slice(self, body: dict) -> dict:
        t0 = time.time()
        tokens = self.resolve_tokens(body)
        use_lens = bool(body.get("use_lens", True))
        top_n = int(body.get("top_n", self.top_n))
        stride = max(1, int(body.get("stride", 1)))
        n_predict = int(body.get("n_predict", 0))
        sampling = body.get("sampling") or {"greedy": True}
        ui_specs = body.get("interventions") or []

        layers = self.readout_layers(stride)
        native_ivs = self.translate_interventions(ui_specs, tokens)

        fr = self.client.forward(
            tokens,
            capture_layers=layers,
            dtype="f16",
            interventions=native_ivs,
            n_predict=n_predict,
            sampling=sampling,
        )
        t_fwd = time.time()

        self._ctx_counter += 1
        ctx_id = f"c{self._ctx_counter}"
        ctx = Context(
            ctx_id, fr.tokens, fr.n_prompt, fr.n_gen, fr.activations, None, use_lens, ui_specs
        )
        # route per-layer logits through the LRU cache so subsequent rank /
        # readout queries on this context reuse them instead of re-running GEMMs
        grid = compute_grid(
            self.readout, fr.activations, layers, top_n=top_n, use_lens=use_lens,
            logits_fn=lambda layer: self.lens_logits_for(ctx, layer),
        )
        ctx.grid = grid
        t_grid = time.time()
        self.contexts[ctx_id] = ctx
        while len(self.contexts) > self.max_contexts:
            old_id, _ = self.contexts.popitem(last=False)
            self.logits_cache.drop_ctx(old_id)

        return {
            "ctx_id": ctx_id,
            "tokens": ctx.tokens,
            "pieces": self.pieces(ctx.tokens),
            "n_prompt": ctx.n_prompt,
            "n_gen": ctx.n_gen,
            "generated_text": fr.generated_text,
            "layers": layers,
            "top_n": top_n,
            "use_lens": use_lens,
            "top_ids": _b64(grid.top_ids.astype("<i4")),
            "norms": {str(l): v for l, v in grid.norms.items()},
            "vocab_size": self.weights.n_vocab,
            "interventions": ui_specs,
            "timings": {
                "forward_ms": round((t_fwd - t0) * 1000, 1),
                "grid_ms": round((t_grid - t_fwd) * 1000, 1),
                **fr.timings,
            },
        }

    def _ctx(self, body: dict) -> Context:
        ctx = self.contexts.get(str(body.get("ctx_id")))
        if ctx is None:
            raise ValueError("unknown or expired ctx_id; re-run the slice")
        return ctx

    def api_ranks(self, body: dict) -> dict:
        ctx = self._ctx(body)
        token_ids = np.asarray([int(t) for t in body["token_ids"]], dtype=np.int64)
        T = len(ctx.tokens)
        L = len(ctx.grid.layers)
        out = np.zeros((T, L, len(token_ids)), dtype=np.int32)
        for li, layer in enumerate(ctx.grid.layers):
            logits = self.lens_logits_for(ctx, layer)
            out[:, li, :] = LensReadout.ranks_of(logits, token_ids)
        return {
            "ctx_id": ctx.ctx_id,
            "token_ids": [int(t) for t in token_ids],
            "shape": [T, L, len(token_ids)],
            "ranks": _b64(out.astype("<i4")),
        }

    def api_readout(self, body: dict) -> dict:
        ctx = self._ctx(body)
        pos = int(body["pos"])
        layer = int(body["layer"])
        top_n = int(body.get("top_n", 40))
        logits = self.lens_logits_for(ctx, layer)[pos]
        ids, vals = LensReadout.topk(logits[None, :], top_n)
        ids, vals = ids[0], vals[0]
        z = logits - logits.max()
        probs_all = np.exp(z)
        probs_all /= probs_all.sum()
        return {
            "pos": pos,
            "layer": layer,
            "tokens": [
                {
                    "token": int(t),
                    "piece": self.vocab[int(t)],
                    "logit": float(v),
                    "prob": float(probs_all[int(t)]),
                }
                for t, v in zip(ids, vals)
            ],
        }

    def api_decompose(self, body: dict) -> dict:
        ctx = self._ctx(body)
        pos = int(body["pos"])
        layer = int(body["layer"])
        k = min(int(body.get("k", 12)), 25)
        if layer not in self.lens.jacobians and layer != self.n_layers - 1:
            raise ValueError(f"layer {layer} not fitted")
        h = ctx.activations[layer][pos]
        items = self.readout.decompose(h, layer, k=k)
        for item in items:
            item["piece"] = self.vocab[item["token"]]
        return {"pos": pos, "layer": layer, "items": items}

    def api_generate(self, body: dict) -> dict:
        if body.get("ctx_id"):
            tokens = self._ctx(body).tokens[: self._ctx(body).n_prompt]
        else:
            tokens = self.resolve_tokens(body)
        n_predict = int(body.get("n_predict", 32))
        sampling = body.get("sampling") or {"greedy": True}
        ui_specs = body.get("interventions") or []
        native_ivs = self.translate_interventions(ui_specs, tokens)

        fr = self.client.forward(
            tokens, capture=False, interventions=native_ivs,
            n_predict=n_predict, sampling=sampling,
        )
        out = {
            "steered": {
                "text": fr.generated_text,
                "tokens": [g["token"] for g in fr.generated],
            }
        }
        if body.get("compare", True) and ui_specs:
            fr0 = self.client.forward(tokens, capture=False, n_predict=n_predict, sampling=sampling)
            out["baseline"] = {
                "text": fr0.generated_text,
                "tokens": [g["token"] for g in fr0.generated],
            }
        return out

    # ------------------------------------------------------------------ #
    # backend mode: steer the OpenAI endpoint apps talk to
    # ------------------------------------------------------------------ #

    def api_live_push(self, body: dict) -> dict:
        """Install the UI's intervention specs as the native server's live
        set, applied to every /v1 completion. Position ranges are dropped to
        all-positions by default: they refer to visualizer prompts, which do
        not line up with an app's conversation tokens."""
        specs = body.get("interventions") or []
        if not body.get("keep_positions", False):
            specs = [{**s, "pos": [0, -1]} for s in specs]
        native = self.translate_interventions(specs, tokens=None)
        out = self.client.live_interventions_set(native, meta={"ui_specs": specs})
        return {"count": out.get("count", 0), "specs": specs}

    def api_live_clear(self, body: dict) -> dict:
        return self.client.live_interventions_clear()

    def api_live_state(self) -> dict:
        live = self.client.live_interventions_get()
        last = self.client.last_completion()
        return {
            "backend_url": self.client.base_url,
            "openai_url": self.client.base_url + "/v1",
            "count": live.get("count", 0),
            "ui_specs": (live.get("meta") or {}).get("ui_specs", []),
            "last_completion": {
                "id": last.get("id", 0),
                "n_prompt": last.get("n_prompt", 0),
                "n_gen": last.get("n_gen", 0),
                "interventions_active": last.get("interventions_active", 0),
            },
        }

    def api_live_last(self) -> dict:
        """The most recent /v1 completion's exact tokens, ready to load as a
        slice (so the visualizer shows the conversation the app just had)."""
        last = self.client.last_completion()
        if not last.get("tokens"):
            raise ValueError("no completion has been served yet")
        return {
            "tokens": last["tokens"],
            "n_prompt": last["n_prompt"],
            "n_gen": last["n_gen"],
            "text": last.get("text", ""),
            "pieces": self.pieces(last["tokens"]),
        }

    def api_search_tokens(self, query: str, limit: int = 50) -> dict:
        q = query.strip()
        results: list[tuple[int, int]] = []  # (score, token_id)
        if q.startswith("#") and q[1:].isdigit():
            tid = int(q[1:])
            if 0 <= tid < len(self.vocab):
                results.append((0, tid))
        ql = q.lower()
        if ql:
            for tid, piece in enumerate(self.vocab_lower):
                stripped = piece.strip()
                if not stripped:
                    continue
                if stripped == ql:
                    results.append((1, tid))
                elif stripped.startswith(ql):
                    results.append((2, tid))
                elif ql in stripped:
                    results.append((3, tid))
        results.sort(key=lambda x: (x[0], len(self.vocab[x[1]]), x[1]))
        out = []
        for _, tid in results[:limit]:
            out.append({"token": tid, "piece": self.vocab[tid]})
        return {"results": out}


# ---------------------------------------------------------------------- #
# HTTP plumbing
# ---------------------------------------------------------------------- #


class Handler(BaseHTTPRequestHandler):
    app: App = None  # set by serve()
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        logger.debug("%s " + fmt, self.address_string(), *args)

    def _send_json(self, obj, status=200):
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _send_file(self, path: Path):
        if not path.is_file():
            self._send_json({"error": "not found"}, 404)
            return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", MIME.get(path.suffix, "application/octet-stream"))
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        try:
            url = urlparse(self.path)
            if url.path == "/api/props":
                return self._send_json(self.app.api_props())
            if url.path == "/api/vocab":
                return self._send_json({"pieces": self.app.vocab, "attrs": self.app.vocab_attrs})
            if url.path == "/api/live/state":
                return self._send_json(self.app.api_live_state())
            if url.path == "/api/live/last":
                return self._send_json(self.app.api_live_last())
            if url.path == "/api/search_tokens":
                qs = parse_qs(url.query)
                q = qs.get("q", [""])[0]
                limit = int(qs.get("limit", ["50"])[0])
                return self._send_json(self.app.api_search_tokens(q, limit))
            # static files (path-boundary checked; see safe_static_path)
            rel = url.path.lstrip("/") or "index.html"
            target = safe_static_path(rel)
            if target is None:
                return self._send_json({"error": "forbidden"}, 403)
            return self._send_file(target)
        except ValueError as e:
            self._send_json({"error": str(e)}, 400)
        except Exception as e:
            logger.exception("GET %s failed", self.path)
            self._send_json({"error": str(e)}, 500)

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            url = urlparse(self.path)
            app = self.app
            routes = {
                "/api/slice": app.api_slice,
                "/api/ranks": app.api_ranks,
                "/api/readout": app.api_readout,
                "/api/decompose": app.api_decompose,
                "/api/generate": app.api_generate,
                "/api/live/push": app.api_live_push,
                "/api/live/clear": app.api_live_clear,
            }
            if url.path == "/api/tokenize":
                toks, pieces = app.client.tokenize_with_pieces(body["content"])
                return self._send_json({"tokens": toks, "pieces": pieces})
            if url.path == "/api/template":
                prompt = app.client.apply_template(
                    body["messages"], add_assistant=bool(body.get("add_assistant", True))
                )
                return self._send_json({"prompt": prompt})
            fn = routes.get(url.path)
            if fn is None:
                return self._send_json({"error": "not found"}, 404)
            with app.lock:
                return self._send_json(fn(body))
        except ValueError as e:
            self._send_json({"error": str(e)}, 400)
        except Exception as e:
            logger.exception("POST %s failed", self.path)
            self._send_json({"error": str(e)}, 500)


def serve(app: App, host: str = "127.0.0.1", port: int = 8090):
    Handler.app = app
    httpd = ThreadingHTTPServer((host, port), Handler)
    logger.info("jlens bridge listening on http://%s:%d", host, port)
    print(f"\n  ── J-Lens visualizer ready: http://{host}:{port} ──\n", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
