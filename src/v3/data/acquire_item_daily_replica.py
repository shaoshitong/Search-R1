#!/usr/bin/env python3
"""Fetch public metadata replica and verify LFS SHA plus official ZIP CRC/size."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
import zlib

URL = "https://hf-mirror.com/hiiamkik/miniOneRec-kuairec/resolve/c644945dfa71b5d6f67ae9c7aff0eae40caf06b8/item_daily_features.csv?download=true"
EXPECTED_BYTES = 85855213
EXPECTED_SHA = "45943d63c44652b6403f3a4f78c7225e1afe7916bab17d9a674d7979245e085b"
OFFICIAL_ZIP_CRC = "12e08ed8"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    partial = args.root / "item_daily_features.replica.csv.partial"
    target = args.root / "item_daily_features.csv"
    observations = []
    started = time.monotonic()
    with (args.root / "replica_download.stderr").open("w") as stderr:
        process = subprocess.Popen(["curl", "-q", "-4", "-fL", "--connect-timeout", "8",
            "--max-time", "600", "--silent", "--show-error", "--output", str(partial), URL], stderr=stderr)
        while process.poll() is None:
            point = {"utc": datetime.now(timezone.utc).isoformat(),
                     "seconds": round(time.monotonic()-started,2),
                     "bytes": partial.stat().st_size if partial.exists() else 0}
            observations.append(point)
            print(json.dumps({"progress": point}), flush=True)
            time.sleep(2)
        code = process.wait()
    digest, crc = hashlib.sha256(), 0
    if partial.exists():
        with partial.open("rb") as stream:
            for block in iter(lambda: stream.read(8*1024*1024), b""):
                digest.update(block)
                crc = zlib.crc32(block, crc)
    size = partial.stat().st_size if partial.exists() else 0
    crc = format(crc & 0xffffffff, "08x")
    verified = code == 0 and size == EXPECTED_BYTES and digest.hexdigest() == EXPECTED_SHA and crc == OFFICIAL_ZIP_CRC
    if verified:
        partial.rename(target)
    receipt = {"replica_url": URL, "replica_public_and_non_gated": True,
               "official_archive_url": "https://zenodo.org/records/18164998/files/KuaiRec.zip",
               "official_member": "KuaiRec 2.0/data/item_daily_features.csv",
               "official_member_bytes": EXPECTED_BYTES, "official_member_crc32": OFFICIAL_ZIP_CRC,
               "replica_lfs_sha256": EXPECTED_SHA, "bytes": size, "sha256": digest.hexdigest(),
               "crc32": crc, "verified": verified, "curl_exit_code": code,
               "whole_official_zip_md5_verified": False,
               "progress": observations, "completed_utc": datetime.now(timezone.utc).isoformat(),
               "path": str(target if verified else partial)}
    (args.root / "item_daily_replica_receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps({"result": receipt}), flush=True)
    if not verified:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
