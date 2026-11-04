#!/usr/bin/env python3
"""Fetch the eight public SeaArt metadata Parquets; verify pinned LFS hashes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

REPO = "lehduong/seaart-hq"
COMMIT = "4321a586faf084ddfa6ab415caae4073af13aa67"
FILES = [
    ("split_1.parquet", 2094835, "9a11e9c594fef3ce75450204924d61885dba4821826a17fe5ca3e653093e8e22"),
    ("split_2.parquet", 2108994, "3fcd28145761f5a3d18f83fb8050d2cadd6771b06313ce7f383ba030ce7e8904"),
    ("split_3.parquet", 2150638, "101e552eae1c98843cfb00d1c27241f4faaf865810e91b1bdbf66de14690cabc"),
    ("split_4.parquet", 2145678, "941623caa837c9b04a03fa50a43f22dbc3b79e8afdde504c9b5dfddf16126354"),
    ("split_5.parquet", 2134954, "a9d22968510e1b50709e1c719853aa7f7e1ab06ced9d8b3354da294364e8309a"),
    ("split_6.parquet", 2144058, "eb4e349324f70014982c3ab03732385345f2f41c58e1c8f3191c3115a67f579b"),
    ("split_7.parquet", 2049275, "8854b4d822671d5930fe53b0942df24b6aea6cf0af1e18d194ee641577ab2395"),
    ("split_8.parquet", 2076251, "75ac5d48a6fe00102e9d2fa28c0e6a6f5fc22c4f0a890ac4962430f01f93832f"),
]


def utc():
    return datetime.now(timezone.utc).isoformat()


def download(entry, raw_root):
    name, expected_bytes, expected_sha = entry
    source = f"https://huggingface.co/datasets/{REPO}/resolve/{COMMIT}/{name}"
    url = source.replace("https://huggingface.co/", "https://hf-mirror.com/")
    path = raw_root / name
    partial = path.with_suffix(".parquet.part")
    if path.exists() and path.stat().st_size == expected_bytes:
        actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_sha == expected_sha:
            print(json.dumps({"file": name, "reused_verified_file": True, "bytes": expected_bytes}), flush=True)
            return {"filename": name, "source_url": source, "download_url": url,
                    "bytes": expected_bytes, "expected_bytes": expected_bytes,
                    "sha256": actual_sha, "expected_lfs_sha256": expected_sha,
                    "verified": True, "reused_verified_file": True}
    observations = []
    started = time.monotonic()
    process = subprocess.Popen([
        "curl", "-q", "-4", "-fL", "--connect-timeout", "8", "--max-time", "90",
        "--retry", "2", "--retry-delay", "1", "--silent", "--show-error",
        "--output", str(partial), "--write-out", "%{http_code}", url,
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    while process.poll() is None:
        size = partial.stat().st_size if partial.exists() else 0
        point = {"utc": utc(), "bytes": size}
        observations.append(point)
        if len(observations) == 1 or size != observations[-2]["bytes"]:
            print(json.dumps({"file": name, "progress": point}), flush=True)
        time.sleep(0.5)
    status, stderr = process.communicate()
    if process.returncode:
        raise RuntimeError(f"curl failed for {name}, exit {process.returncode}: {stderr}")
    size = partial.stat().st_size
    observations.append({"utc": utc(), "bytes": size})
    actual_sha = hashlib.sha256(partial.read_bytes()).hexdigest()
    verified = size == expected_bytes and actual_sha == expected_sha
    receipt = {
        "filename": name, "source_url": source, "download_url": url,
        "http_status": int(status), "curl_exit_code": process.returncode,
        "expected_bytes": expected_bytes, "bytes": size,
        "expected_lfs_sha256": expected_sha, "sha256": actual_sha,
        "verified": verified, "progress": observations,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    if verified:
        partial.replace(path)
    else:
        raise ValueError(f"LFS verification failed: {name}: {receipt}")
    print(json.dumps({"file": name, "verified": True, "bytes": size}), flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="Remote storage root, e.g. /data/.../v3/seaart")
    args = parser.parse_args()
    raw_root = args.root / "raw"
    audit_root = args.root / "audit"
    raw_root.mkdir(parents=True, exist_ok=True)
    audit_root.mkdir(parents=True, exist_ok=True)
    started = utc()
    with ThreadPoolExecutor(max_workers=4) as pool:
        receipts = list(pool.map(lambda entry: download(entry, raw_root), FILES))
    receipt = {
        "repository": REPO, "commit": COMMIT, "public_non_gated": True,
        "license_declared_by_source": None, "record_type": "metadata only; no media fetched",
        "raw_root": str(raw_root), "started_utc": started, "completed_utc": utc(),
        "file_count": len(receipts), "total_bytes": sum(r["bytes"] for r in receipts),
        "all_lfs_sha256_verified": all(r["verified"] for r in receipts),
        "files": receipts,
    }
    (audit_root / "seaart_download_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"completed": True, "file_count": len(receipts),
                      "total_bytes": receipt["total_bytes"], "all_lfs_sha256_verified": True}), flush=True)


if __name__ == "__main__":
    main()
