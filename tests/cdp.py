# Minimal Chrome DevTools Protocol driver for UI testing.
import base64
import json
import subprocess
import time
import urllib.request

import websocket


class Browser:
    def __init__(self, port=9777, window="1500,950"):
        self.port = port
        self.proc = subprocess.Popen(
            [
                "chromium-browser", "--headless=new", "--disable-gpu", "--no-sandbox",
                f"--remote-debugging-port={port}", f"--window-size={window}",
                "--remote-allow-origins=*", "about:blank",
            ],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self.ws = None
        self._id = 0
        for _ in range(80):
            try:
                targets = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json"))
                page = next(t for t in targets if t["type"] == "page")
                self.ws = websocket.create_connection(page["webSocketDebuggerUrl"], timeout=60)
                break
            except Exception:
                time.sleep(0.25)
        if self.ws is None:
            raise RuntimeError("could not connect to chromium CDP")
        self.cmd("Page.enable")
        self.cmd("Runtime.enable")
        self.console = []

    def cmd(self, method, **params):
        self._id += 1
        self.ws.send(json.dumps({"id": self._id, "method": method, "params": params}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("method") == "Runtime.consoleAPICalled":
                args = [a.get("value", a.get("description", "")) for a in msg["params"]["args"]]
                self.console.append((msg["params"]["type"], " ".join(str(a) for a in args)))
                continue
            if msg.get("method") == "Runtime.exceptionThrown":
                exc = msg["params"]["exceptionDetails"]
                self.console.append(("exception", json.dumps(exc.get("exception", {}).get("description", exc))))
                continue
            if msg.get("id") == self._id:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})

    def goto(self, url):
        self.cmd("Page.navigate", url=url)
        time.sleep(1.0)

    def js(self, expression, await_promise=False):
        res = self.cmd(
            "Runtime.evaluate", expression=expression,
            awaitPromise=await_promise, returnByValue=True, timeout=60000,
        )
        if "exceptionDetails" in res:
            desc = res["exceptionDetails"].get("exception", {}).get("description", str(res["exceptionDetails"]))
            raise RuntimeError(f"JS exception: {desc}")
        return res.get("result", {}).get("value")

    def wait_for(self, expression, timeout=30.0, interval=0.25):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.js(expression):
                return True
            time.sleep(interval)
        raise TimeoutError(f"condition never true: {expression}")

    def screenshot(self, path):
        data = self.cmd("Page.captureScreenshot", format="png")["data"]
        with open(path, "wb") as f:
            f.write(base64.b64decode(data))

    def errors(self):
        return [c for c in self.console if c[0] in ("error", "exception")]

    def close(self):
        try:
            self.ws.close()
        finally:
            self.proc.terminate()
            self.proc.wait(timeout=10)
