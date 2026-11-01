#!/usr/bin/env python3
"""Explicitly download/hash-check official Paddle models and export to ONNX.
Requires Python 3.12+, paddlepaddle==3.3.1, paddle2onnx==2.1.0, onnx==1.17.0.
Weights are not distributed with RustO. No package installs run from this script.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import urllib.request

MODELS = [
    ("SLANet_plus", "slanet-plus.onnx", 7997440, "68526c5c6573a8a17cdc14805e026f11c2545e42d89a1317bb58d3e1e4b58318"),
    ("PP-DocLayout-S", "doclayout-s.onnx", 5150720, "3f589aa473a5305a626705d94762fcb4ab3e43e6e48983c9b34b011a6c9d0394"),
]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--proxy", help="Optional HTTP proxy, e.g. http://127.0.0.1:1087")
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    handlers = [urllib.request.ProxyHandler({"http": args.proxy, "https": args.proxy})] if args.proxy else []
    opener = urllib.request.build_opener(*handlers)
    records = []
    for name, converted_name, size, digest in MODELS:
        url = f"https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/{name}_infer.tar"
        archive = args.output / f"{name}_infer.tar"
        if not archive.exists() or sha(archive) != digest:
            partial = archive.with_suffix(".tar.partial")
            with opener.open(url, timeout=60) as source, partial.open("wb") as target:
                total = 0
                while chunk := source.read(1024 * 1024):
                    total += len(chunk)
                    if total > size:
                        raise ValueError(f"{name}: download exceeds pinned size")
                    target.write(chunk)
            if partial.stat().st_size != size or sha(partial) != digest:
                raise ValueError(f"{name}: source hash/size mismatch")
            partial.replace(archive)
        with tarfile.open(archive) as source:
            source.extractall(args.output, filter="data")
        record = {"name": name, "source": url, "source_sha256": digest}
        if not args.download_only:
            output = args.output / converted_name
            temporary = output.with_suffix(".partial.onnx")
            subprocess.run([
                "paddle2onnx", "--model_dir", str(args.output / f"{name}_infer"),
                "--model_filename", "inference.json", "--params_filename", "inference.pdiparams",
                "--save_file", str(temporary), "--opset_version", "17",
                "--enable_onnx_checker", "True", "--optimize_tool", "None",
            ], check=True, env={**os.environ, "POLYGRAPHY_AUTOINSTALL_DEPS": "0"})
            temporary.replace(output)
            record.update(onnx_file=converted_name, onnx_sha256=sha(output), onnx_bytes=output.stat().st_size)
        records.append(record)
    (args.output / "table-model-manifest.json").write_text(json.dumps(records, indent=2) + "\n")

if __name__ == "__main__":
    main()
