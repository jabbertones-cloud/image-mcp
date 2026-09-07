"""Drive run_server.py over real stdio the way an MCP client does, for the AI paths.

    <python> tools\\smoke_stdio.py [--python <interpreter>] [sd|qwen|llada|status]...

Default: status. ``sd`` runs sd_status -> sd_generate (Lykon/dreamshaper-8, 512px, 12 steps - SD1.5 below 512px is mush)
-> sd_img2img -> sd_inpaint and saves the results under %TEMP%\\imagetools_smoke.
Fails loudly if anything other than JSON-RPC appears on stdout or a call takes
longer than the per-call deadline.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("TEMP", ".")) / "imagetools_smoke"


class Client:
    def __init__(self, python: str):
        self.proc = subprocess.Popen(
            [python, str(ROOT / "run_server.py")],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8",
        )
        self.err: list[str] = []
        threading.Thread(target=lambda: self.err.extend(self.proc.stderr), daemon=True).start()
        self._id = 0

    def _send(self, obj: dict) -> None:
        self.proc.stdin.write(json.dumps(obj) + "\n")
        self.proc.stdin.flush()

    def request(self, method: str, params: dict, timeout: float = 60.0) -> dict:
        self._id += 1
        rid = self._id
        self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            line = self.proc.stdout.readline()
            if not line:
                raise SystemExit(f"server closed stdout during {method}; stderr tail:\n{''.join(self.err)[-3000:]}")
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                raise SystemExit(f"NON-JSON ON STDOUT: {line[:200]!r}")
            if obj.get("id") == rid:
                return obj
        raise SystemExit(f"TIMEOUT after {timeout:.0f}s waiting for {method} {params.get('name', '')}")

    def call(self, name: str, timeout: float = 120.0, **args) -> dict:
        t0 = time.perf_counter()
        rep = self.request("tools/call", {"name": name, "arguments": args}, timeout=timeout)
        dt = time.perf_counter() - t0
        if "error" in rep:
            raise SystemExit(f"{name}: JSON-RPC error {rep['error']}")
        res = rep["result"]
        text = "".join(c.get("text", "") for c in res.get("content", []))
        if res.get("isError"):
            raise SystemExit(f"{name}: isError after {dt:.1f}s: {text[:800]}")
        try:
            data = json.loads(text) if text.startswith("{") else {"text": text}
        except json.JSONDecodeError:
            data = {"text": text}
        print(f"  {name:22} {dt:7.1f}s  {json.dumps(data)[:160]}")
        return data

    def close(self) -> int:
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        return self.proc.returncode


def run_sd(c: Client) -> None:
    c.call("sd_status", timeout=900)
    c.call("sd_generate", timeout=900, prompt="a lighthouse on a rocky coast at sunset", model="Lykon/dreamshaper-8",
           width=512, height=512, steps=12, seed=7, canvas_id="smoke-sd")
    c.call("save_canvas", canvas_id="smoke-sd", path=str(OUT / "sd_txt2img.png"))
    c.call("sd_img2img", timeout=900, canvas_id="smoke-sd", prompt="the same lighthouse in a snowstorm",
           strength=0.6, steps=8, seed=7, new_canvas_id="smoke-sd-i2i")
    c.call("save_canvas", canvas_id="smoke-sd-i2i", path=str(OUT / "sd_img2img.png"))
    c.call("new_canvas", width=512, height=512, color="black", canvas_id="smoke-mask")
    c.call("draw_rectangle", canvas_id="smoke-mask", x1=128, y1=128, x2=384, y2=384, fill="white")
    c.call("sd_inpaint", timeout=900, canvas_id="smoke-sd", mask_canvas_id="smoke-mask",
           prompt="a red hot air balloon", steps=8, seed=7, new_canvas_id="smoke-sd-inp")
    c.call("save_canvas", canvas_id="smoke-sd-inp", path=str(OUT / "sd_inpaint.png"))
    c.call("sd_unload")


def main(argv: list[str]) -> int:
    python = sys.executable
    if "--python" in argv:
        i = argv.index("--python")
        python = argv[i + 1]
        del argv[i:i + 2]
    groups = argv or ["status"]
    OUT.mkdir(parents=True, exist_ok=True)
    c = Client(python)
    ok = True
    try:
        init = c.request("initialize", {"protocolVersion": "2025-03-26", "capabilities": {},
                                        "clientInfo": {"name": "smoke", "version": "1"}}, timeout=120)
        print("initialize:", init["result"]["serverInfo"])
        c._send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        tools = c.request("tools/list", {}, timeout=60)["result"]["tools"]
        print("tools/list:", len(tools))
        for g in groups:
            print(f"--- {g}")
            if g == "status":
                c.call("sd_status", timeout=900)
                c.call("qwen_status", timeout=900)
            elif g == "sd":
                run_sd(c)
            elif g == "llada":
                c.call("llada_status", timeout=900)
                c.call("llada_load", timeout=1800, model="turbo")
                c.call("llada_generate", timeout=1800, prompt="a red fox standing in fresh snow, soft winter light",
                       width=512, height=512, seed=42, canvas_id="smoke-llada")
                c.call("save_canvas", canvas_id="smoke-llada", path=str(OUT / "llada_txt2img.png"))
                c.call("llada_edit", timeout=1800, canvas_id="smoke-llada", prompt="turn it into a watercolor painting",
                       seed=43, new_canvas_id="smoke-llada-edit")
                c.call("save_canvas", canvas_id="smoke-llada-edit", path=str(OUT / "llada_edit.png"))
                c.call("llada_status", timeout=900)
                c.call("llada_unload", timeout=900)
            else:
                raise SystemExit(f"unknown group {g}; valid: status, sd, llada")
    except SystemExit as e:
        print("FAIL:", e)
        ok = False
    finally:
        rc = c.close()
        print(f"server exit code: {rc}")
        tb = [l for l in c.err if "Traceback" in l]
        if tb:
            print("stderr tracebacks:\n", "".join(c.err)[-2500:])
            ok = False
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
