#!/usr/bin/env python3
"""Download the public, pinned Civitai metadata snapshot on an HKUST host."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
from datetime import datetime, timezone

REPO = "thefcraft/civitai-stable-diffusion-337k"
COMMIT = "758d4d72c87c5177c349230f38f2e8e8e3b6a23c"
NAME = "train-00000-of-00001-ace5b28cebba25a7.parquet"
EXPECTED_BYTES = 164627773
EXPECTED_SHA256 = "f6e0ce6f3c721a3c5ecff187681c530d3df8e644656823b73d478efc06c07a2e"
URL = f"https://hf-mirror.com/datasets/{REPO}/resolve/{COMMIT}/data/{NAME}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    raw = args.root / "raw"
    audit = args.root / "audit"
    raw.mkdir(parents=True, exist_ok=True)
    audit.mkdir(parents=True, exist_ok=True)
    partial = raw / f"{NAME}.partial"
    final = raw / NAME
    started = time.monotonic()
    observations = []
    with (audit / "download.stderr").open("w") as stderr:
        process = subprocess.Popen(
            ["curl", "-4", "-fL", "--connect-timeout", "8", "--max-time", "600",
             "--speed-limit", "1024", "--speed-time", "60", "--silent", "--show-error",
             "--output", str(partial), URL], stderr=stderr,
        )
        while process.poll() is None:
            point = {"elapsed_seconds": round(time.monotonic() - started, 2),
                     "bytes": partial.stat().st_size if partial.exists() else 0}
            observations.append(point)
            print(json.dumps({"download_progress": point}), flush=True)
            time.sleep(2)
        code = process.wait()
    size = partial.stat().st_size if partial.exists() else 0
    sha = None
    if partial.exists():
        digest = hashlib.sha256()
        with partial.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        sha = digest.hexdigest()
    verified = code == 0 and size == EXPECTED_BYTES and sha == EXPECTED_SHA256
    if verified:
        partial.rename(final)
    receipt = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "repository": REPO, "source_commit": COMMIT, "mirror_url": URL,
        "gated": False, "private": False, "license": "not declared in dataset card metadata",
        "expected_bytes": EXPECTED_BYTES, "bytes": size,
        "expected_sha256": EXPECTED_SHA256, "sha256": sha,
        "curl_exit_code": code, "verified": verified,
        "path": str(final if verified else partial), "progress": observations,
        "images_downloaded": False,
    }
    (audit / "download_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"download_result": receipt}), flush=True)
    if not verified:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
