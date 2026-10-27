"""Start a vLLM OpenAI-compatible server for one base model and its adapters, then hold it open.

    python serve/server.py --base ibm-granite/granite-4.1-8b --out $DATA_ROOT/serve/granite-4.1-8b \
        [--adapter clean=/path/to/adapter ...] [--max-model-len 32768] [--gpu-mem 0.85] [--tp 1]

Writes <out>/endpoint.json (base_url, base, adapters, pid) once /v1/models answers; runners read it.
Stops the server on SIGINT/SIGTERM. The job that runs this must stay alive for as long as the server is needed.
"""

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return int(s.getsockname()[1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="hub id or local folder")
    ap.add_argument("--out", required=True)
    ap.add_argument("--adapter", action="append", default=[], help="name=path of a LoRA adapter")
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--max-lora-rank", type=int, default=32)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--startup-timeout", type=int, default=1200)
    args = ap.parse_args()

    port = args.port or free_port()
    vllm_bin = os.path.join(os.path.dirname(sys.executable), "vllm")  # the venv's own, since its bin is not on PATH
    cmd = [
        vllm_bin, "serve", args.base,
        "--host", "127.0.0.1", "--port", str(port),
        "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", str(args.gpu_mem),
        "--tensor-parallel-size", str(args.tp),
        "--enable-prefix-caching",
        "--enable-prompt-tokens-details",  # else usage.prompt_tokens_details.cached_tokens is never reported
    ]  # fmt: skip
    if args.adapter:
        cmd += ["--enable-lora", "--max-lora-rank", str(args.max_lora_rank), "--lora-modules", *args.adapter]
    os.makedirs(args.out, exist_ok=True)
    log = open(os.path.join(args.out, "vllm.log"), "a")
    print(" ".join(cmd), file=log, flush=True)
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)

    def stop(*_):
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    base_url = f"http://127.0.0.1:{port}/v1"
    t0 = time.time()
    while True:
        if proc.poll() is not None:
            raise SystemExit(f"vllm exited with {proc.returncode}; see {log.name}")
        try:
            with urllib.request.urlopen(f"{base_url}/models", timeout=5) as r:
                served = [m["id"] for m in json.load(r)["data"]]
                break
        except Exception:  # noqa: BLE001 - not up yet
            if time.time() - t0 > args.startup_timeout:
                stop()
            time.sleep(5)
    with open(os.path.join(args.out, "endpoint.json"), "w") as f:
        json.dump(
            {"base_url": base_url, "base": args.base, "served": served, "adapters": args.adapter, "pid": proc.pid, "host": socket.gethostname()},
            f,
            indent=2,
        )
    print(f"serving {served} at {base_url} after {time.time() - t0:.0f}s; endpoint.json in {args.out}", flush=True)
    proc.wait()


if __name__ == "__main__":
    main()
