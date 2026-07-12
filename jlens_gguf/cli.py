# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
"""jlens_gguf command line.

    python -m jlens_gguf serve   --model m.gguf [--lens lens.gguf]
    python -m jlens_gguf fit     --model m.gguf --corpus wikitext:100 -o lens.gguf
    python -m jlens_gguf convert-pt lens.pt lens.gguf
    python -m jlens_gguf identity --model m.gguf -o identity.gguf
    python -m jlens_gguf inspect lens.gguf

``serve`` and ``fit`` need the native activation server (jlens-server). By
default they look for one at --native-url and, failing that, start one
themselves from --native-bin (built by native/build.sh).
"""

from __future__ import annotations

import argparse
import atexit
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger("jlens_gguf")


def _find_native_bin(explicit: str | None) -> str | None:
    candidates = [
        explicit,
        os.environ.get("JLENS_NATIVE_BIN"),
        str(Path(__file__).resolve().parent.parent / "native" / "jlens-server"),
        shutil.which("jlens-server"),
    ]
    for c in candidates:
        if c and Path(c).is_file() and os.access(c, os.X_OK):
            return c
    return None


def _model_from_llama_server(url: str) -> str | None:
    """Read the GGUF path a running llama-server / jlens-server is serving,
    from its ``/props`` (``model_path``). Returns None if unreachable or the
    path isn't local."""
    import requests

    for path in ("/props", "/v1/props"):
        try:
            r = requests.get(url.rstrip("/") + path, timeout=5)
            if r.status_code != 200:
                continue
            data = r.json()
            model_path = data.get("model_path") or (
                data.get("default_generation_settings", {}) or {}
            ).get("model", {}).get("path")
            if model_path and model_path != "none" and os.path.exists(model_path):
                return model_path
            if model_path and model_path != "none":
                logger.warning(
                    "llama-server reports model_path=%r but it is not accessible "
                    "from here; pass --model explicitly", model_path
                )
        except Exception:
            continue
    return None


def _ensure_native(args) -> str:
    """Return a live native-server URL, spawning jlens-server if needed."""
    from jlens_gguf.client import NativeClient

    # quickstart: discover the model a running llama-server is using
    if getattr(args, "llama_server", None) and not args.model:
        model = _model_from_llama_server(args.llama_server)
        if model:
            logger.info("discovered model from %s: %s", args.llama_server, model)
            args.model = model
        else:
            sys.exit(
                f"error: could not read a local model path from {args.llama_server}. "
                "Is it a llama-server with a local GGUF? Pass --model explicitly."
            )

    url = args.native_url
    client = NativeClient(url)
    if client.health():
        props = client.props()
        logger.info("using running jlens-server at %s (%s)", url, props.get("model_desc"))
        if args.model and Path(props.get("model_path", "")).resolve() != Path(args.model).resolve():
            logger.warning(
                "native server model (%s) differs from --model (%s); "
                "readout weights must match the served model!",
                props.get("model_path"), args.model,
            )
        return url

    if not args.model:
        sys.exit(f"error: no jlens-server at {url} and no --model given to start one")
    native_bin = _find_native_bin(args.native_bin)
    if native_bin is None:
        sys.exit(
            "error: jlens-server binary not found. Build it first:\n"
            "    LLAMA_DIR=/path/to/llama.cpp native/build.sh\n"
            "or pass --native-bin / set JLENS_NATIVE_BIN."
        )
    port = int(url.rsplit(":", 1)[-1].split("/")[0])
    cmd = [
        native_bin, "-m", args.model,
        "--port", str(port),
        "-c", str(args.ctx_size),
        "--chunk", str(args.chunk),
    ]
    if args.threads:
        cmd += ["-t", str(args.threads)]
    if args.n_gpu_layers:
        cmd += ["--n-gpu-layers", str(args.n_gpu_layers)]
    logger.info("starting native server: %s", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    atexit.register(proc.terminate)

    for _ in range(600):
        if client.health():
            return url
        if proc.poll() is not None:
            sys.exit(f"error: jlens-server exited with code {proc.returncode}")
        time.sleep(0.25)
    sys.exit("error: jlens-server did not become healthy in 150s")


def _default_lens_for(model_path: str) -> str | None:
    """A lens next to the model or in ./lenses that matches the model name,
    so quickstart can auto-load a fitted lens when one exists."""
    model = Path(model_path)
    stem = model.stem
    candidates = [
        model.with_suffix(".jlens.gguf"),
        model.parent / f"{stem}-jlens.gguf",
        Path("lenses") / f"{stem}-jlens-reg.gguf",
        Path("lenses") / f"{stem}.gguf",
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    return None


def cmd_serve(args) -> None:
    from jlens_gguf.client import NativeClient
    from jlens_gguf.server import App, serve

    url = _ensure_native(args)
    if not args.model:
        # a native server was already running and no --model was given: the
        # bridge still needs the GGUF to read readout weights (final norm +
        # unembedding). Take the path the native server reports.
        model_path = NativeClient(url).props().get("model_path")
        if not model_path or not os.path.exists(model_path):
            sys.exit(
                f"error: jlens-server at {url} serves {model_path!r}, which is not "
                "accessible here; pass --model with the local GGUF path."
            )
        args.model = model_path
        logger.info("using model reported by native server: %s", model_path)
    lens = args.lens
    if lens is None and getattr(args, "auto_lens", False):
        lens = _default_lens_for(args.model)
        if lens:
            logger.info("auto-loaded lens: %s", lens)
    app = App(
        model_path=args.model,
        native_url=url,
        lens_path=lens,
        top_n=args.top_n,
        max_contexts=args.max_contexts,
        logits_cache_mb=args.logits_cache_mb,
    )
    serve(app, host=args.host, port=args.port)


def cmd_quickstart(args) -> None:
    """The easy path: point at a running llama-server (or a model file) and go.

    Discovers the model from a running llama-server's /props when given
    --llama-server, auto-loads a matching lens if one exists, and opens the
    browser."""
    args.auto_lens = args.lens is None
    if not args.model and not args.llama_server:
        # nothing specified: try the conventional local llama-server
        args.llama_server = "http://127.0.0.1:8080"
        logger.info("no --model/--llama-server given; trying %s", args.llama_server)
    if args.open:
        import threading
        import time
        import webbrowser

        def _open():
            time.sleep(2.5)
            webbrowser.open(f"http://{args.host}:{args.port}")

        threading.Thread(target=_open, daemon=True).start()
    cmd_serve(args)


def cmd_fit(args) -> None:
    from jlens_gguf.fitting import fit_regression, load_corpus

    url = _ensure_native(args)
    from jlens_gguf.client import NativeClient

    client = NativeClient(url)
    prompts = load_corpus(args.corpus, n_prompts=args.n_prompts)
    if not prompts:
        sys.exit(f"error: no usable prompts in corpus {args.corpus!r}")
    logger.info("fitting on %d prompts ...", len(prompts))
    layers = None
    if args.layers:
        layers = [int(x) for x in args.layers.split(",")]
    lens = fit_regression(
        client,
        prompts,
        source_layers=layers,
        target_layer=args.target_layer,
        max_seq_len=args.max_seq_len,
        ridge=args.ridge,
        affine=not args.no_affine,
        base_model=args.model or "",
    )
    lens.save(args.output)
    print(f"saved {lens!r} -> {args.output}")


def cmd_convert_pt(args) -> None:
    from jlens_gguf.pt_convert import convert_pt_lens

    lens = convert_pt_lens(args.input, args.output, base_model=args.base_model)
    print(f"converted {lens!r} -> {args.output}")


def cmd_identity(args) -> None:
    from jlens_gguf.lens import JacobianLensGGUF
    from jlens_gguf.model_reader import ReadoutWeights

    w = ReadoutWeights.from_gguf(args.model)
    lens = JacobianLensGGUF.identity(
        d_model=w.d_model, layers=list(range(w.n_layers - 1)), base_model=args.model
    )
    lens.save(args.output)
    print(f"saved {lens!r} -> {args.output}")


def cmd_inspect(args) -> None:
    from jlens_gguf.lens import JacobianLensGGUF

    lens = JacobianLensGGUF.load(args.path)
    print(lens)
    print(f"  target_layer: {lens.target_layer}")
    print(f"  base_model:   {lens.base_model}")
    print(f"  biases:       {sorted(lens.biases) if lens.biases else 'none'}")
    if lens.h_rms:
        vals = ", ".join(f"L{l}={v:.1f}" for l, v in sorted(lens.h_rms.items())[:8])
        print(f"  h_rms:        {vals}{' ...' if len(lens.h_rms) > 8 else ''}")


def _add_native_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--native-url", default="http://127.0.0.1:8091", help="jlens-server URL")
    p.add_argument("--native-bin", default=None, help="path to jlens-server binary (for autostart)")
    p.add_argument("--ctx-size", type=int, default=4096, help="context size for autostarted server")
    p.add_argument("--chunk", type=int, default=512, help="prompt chunk for autostarted server")
    p.add_argument("--threads", type=int, default=0, help="threads for autostarted server")
    p.add_argument("--n-gpu-layers", type=int, default=0, help="GPU layers for autostarted server")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="jlens_gguf", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("serve", help="run the interactive visualizer")
    p.add_argument("--model", default=None, help="model GGUF path (or use --llama-server)")
    p.add_argument("--lens", default=None, help="lens GGUF (default: identity / logit lens)")
    p.add_argument("--llama-server", default=None,
                   help="URL of a running llama-server; read its model from /props")
    p.add_argument("--auto-lens", action="store_true",
                   help="auto-load a lens matching the model name if one exists")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--max-contexts", type=int, default=3)
    p.add_argument("--logits-cache-mb", type=int, default=1024)
    _add_native_args(p)
    p.set_defaults(fn=cmd_serve)

    p = sub.add_parser(
        "quickstart",
        help="easiest path: point at a running llama-server (or a model) and open the UI",
    )
    p.add_argument("model", nargs="?", default=None,
                   help="model GGUF path (optional if --llama-server is given)")
    p.add_argument("--llama-server", default=None,
                   help="URL of a running llama-server to read the model from "
                        "(default: http://127.0.0.1:8080 if no model given)")
    p.add_argument("--lens", default=None, help="lens GGUF (default: auto-detect, else logit lens)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--max-contexts", type=int, default=3)
    p.add_argument("--logits-cache-mb", type=int, default=1024)
    p.add_argument("--no-open", dest="open", action="store_false", help="don't open a browser")
    _add_native_args(p)
    p.set_defaults(fn=cmd_quickstart)

    p = sub.add_parser("fit", help="fit a regression lens via jlens-server")
    p.add_argument("--model", default=None, help="model GGUF (to autostart the native server)")
    p.add_argument("--corpus", default="wikitext:100",
                   help="'wikitext[:N]' or a local text file (default wikitext:100)")
    p.add_argument("--n-prompts", type=int, default=100)
    p.add_argument("--layers", default=None, help="comma-separated source layers (default: all)")
    p.add_argument("--target-layer", type=int, default=None)
    p.add_argument("--max-seq-len", type=int, default=128)
    p.add_argument("--ridge", type=float, default=1e-4)
    p.add_argument("--no-affine", action="store_true", help="fit without a bias term")
    p.add_argument("-o", "--output", required=True)
    _add_native_args(p)
    p.set_defaults(fn=cmd_fit)

    p = sub.add_parser("convert-pt", help="convert a reference PyTorch lens to GGUF")
    p.add_argument("input", help="lens .pt from the reference implementation")
    p.add_argument("output", help="output .gguf")
    p.add_argument("--base-model", default="")
    p.set_defaults(fn=cmd_convert_pt)

    p = sub.add_parser("identity", help="write an identity (logit-lens) lens for a model")
    p.add_argument("--model", required=True)
    p.add_argument("-o", "--output", required=True)
    p.set_defaults(fn=cmd_identity)

    p = sub.add_parser("inspect", help="print lens GGUF metadata")
    p.add_argument("path")
    p.set_defaults(fn=cmd_inspect)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    args.fn(args)


if __name__ == "__main__":
    main()
