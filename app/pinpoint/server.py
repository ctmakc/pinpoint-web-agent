"""Long-running process: holds the model on the GPU, answers requests over a unix socket.

Each question arrives as a fresh `python3 /app/app.py ...` process, so the model must not be loaded there.
"""
from __future__ import annotations

import json
import os
import socket
import socketserver
import sys
import threading
import time

SOCK = os.environ.get("PINPOINT_SOCK", "/tmp/pinpoint.sock")
READY = SOCK + ".ready"


def main():
    from .agent import Agent
    from .llm import make_llm

    t0 = time.time()
    llm = make_llm()
    # warm-up: first generate call compiles kernels
    try:
        llm.chat("Reply with {}", "ping", max_new_tokens=4)
    except Exception as e:
        print("warm-up failed:", e, file=sys.stderr)
    agent = Agent(llm, budget_s=float(os.environ.get("PINPOINT_BUDGET_S", "24")))
    lock = threading.Lock()  # one generate at a time on the GPU
    info = {"model": llm.name, "gpu": llm.gpu, "hip": llm.hip, "load_s": llm.load_s,
            "ready_s": round(time.time() - t0, 1)}
    print("pinpoint ready", json.dumps(info), flush=True)

    class LockedLLM:
        def __init__(self, inner):
            self.inner = inner
            self.name = inner.name

        def chat(self, *a, **k):
            with lock:
                return self.inner.chat(*a, **k)

    agent.llm = LockedLLM(llm)

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            line = self.rfile.readline()
            try:
                req = json.loads(line)
                if req.get("op") == "info":
                    out = dict(info, vram_peak_gib=llm.vram_gib())
                else:
                    r = agent.run(req["question"], req.get("url"))
                    out = r.to_json()
                    out["vram_peak_gib"] = llm.vram_gib()
            except Exception as e:  # never kill the server on one bad request
                out = {"answer": "", "sources": [], "citations": [], "error": f"{type(e).__name__}: {e}"}
            self.wfile.write((json.dumps(out) + "\n").encode())

    class Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
        daemon_threads = True

    if os.path.exists(SOCK):
        os.unlink(SOCK)
    srv = Server(SOCK, Handler)
    os.chmod(SOCK, 0o777)
    open(READY, "w").write(json.dumps(info))
    srv.serve_forever()


def ask(payload: dict, timeout: float = 60.0) -> dict:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    s.connect(SOCK)
    s.sendall((json.dumps(payload) + "\n").encode())
    buf = b""
    while not buf.endswith(b"\n"):
        chunk = s.recv(65536)
        if not chunk:
            break
        buf += chunk
    s.close()
    return json.loads(buf)


if __name__ == "__main__":
    main()
