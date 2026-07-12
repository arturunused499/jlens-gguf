"""Headless-browser test of the web UI (skipped if chromium is unavailable)."""

import shutil
import socket
import threading
import time

import pytest

CHROMIUM = shutil.which("chromium-browser") or shutil.which("chromium") or shutil.which("google-chrome")


@pytest.fixture(scope="module")
def bridge_url(app):
    from http.server import ThreadingHTTPServer

    from jlens_gguf.server import Handler

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    Handler.app = app
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    httpd.shutdown()


@pytest.mark.skipif(CHROMIUM is None, reason="no chromium available")
def test_ui_end_to_end(bridge_url):
    from cdp import Browser

    b = Browser(port=9977)
    try:
        b.goto(bridge_url + "/?autorun=1&prompt=Once%20upon%20a%20time%20there%20was%20a")
        b.wait_for("typeof D !== 'undefined' && D && D.topIds.length > 0", 40)
        assert b.js("D.L") == 5
        assert b.js("D.T") >= 6

        # click-equivalent: pin the final-layer top-1 and wait for its ranks
        b.js("selCtx = D.T-1; selLayer = D.layers[D.L-1]; pinToken(argmaxAt(D.T-1, D.L-1));")
        b.wait_for("rankCache.size >= 1", 20)

        # add a steer intervention through the modal path
        b.js("openIvModal('steer')")
        tid = b.js("api('/api/search_tokens?q=a&limit=1').then(r=>r.results[0].token)", await_promise=True)
        b.js(f"modalCtl.w1.get = () => {tid}; saveIvModal();")
        b.wait_for("interventions.length === 1 && typeof D !== 'undefined' && D.hadInterventions && !running", 40)
        b.wait_for("currentBaseline() !== null", 40)

        # cell readout panel populated
        b.js("refreshCellPanel()")
        b.wait_for("document.querySelectorAll('#cellReadout .cr-row').length > 5", 20)

        errors = b.errors()
        assert not errors, f"console errors: {errors}"
    finally:
        b.close()
