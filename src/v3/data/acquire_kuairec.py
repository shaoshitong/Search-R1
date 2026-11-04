#!/usr/bin/env python3
"""Acquire the official open KuaiRec metadata archive on HKUST3."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import zipfile

URL = "https://zenodo.org/records/18164998/files/KuaiRec.zip"
EXPECTED_BYTES = 431964858
EXPECTED_MD5 = "261550d472c48eff4990fb13c0e5bcf7"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    partial = args.root / "KuaiRec.zip.partial"
    final = args.root / "KuaiRec.zip"
    started = time.monotonic()
    observations = []
    with (args.root / "download.stderr").open("w") as stderr:
        process = subprocess.Popen(
            ["curl", "-q", "-4", "-fL", "--connect-timeout", "8", "--max-time", "1200",
             "--speed-limit", "1024", "--speed-time", "60", "--silent", "--show-error",
             "--output", str(partial), URL], stderr=stderr)
        while process.poll() is None:
            point = {"utc": datetime.now(timezone.utc).isoformat(),
                     "seconds": round(time.monotonic() - started, 2),
                     "bytes": partial.stat().st_size if partial.exists() else 0}
            observations.append(point)
            print(json.dumps({"progress": point}), flush=True)
            time.sleep(5)
        code = process.wait()
    actual_bytes = partial.stat().st_size if partial.exists() else 0
    md5, sha = hashlib.md5(), hashlib.sha256()
    if partial.exists():
        with partial.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                md5.update(block)
                sha.update(block)
    verified = code == 0 and actual_bytes == EXPECTED_BYTES and md5.hexdigest() == EXPECTED_MD5
    if verified:
        partial.rename(final)
    receipt = {"source": URL, "record_api": "https://zenodo.org/api/records/18164998",
               "doi": "10.5281/zenodo.18164998", "access_right": "open",
               "zenodo_license": "CC-BY-4.0", "github_repository_license": "CC-BY-SA-4.0",
               "expected_bytes": EXPECTED_BYTES, "actual_bytes": actual_bytes,
               "expected_md5": EXPECTED_MD5, "md5": md5.hexdigest(), "sha256": sha.hexdigest(),
               "curl_exit_code": code, "verified": verified, "progress": observations,
               "completed_utc": datetime.now(timezone.utc).isoformat(),
               "archive_path": str(final if verified else partial)}
    (args.root / "download_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"download_result": receipt}), flush=True)
    if not verified:
        raise SystemExit(1)
    with zipfile.ZipFile(final) as archive:
        inventory = [{"name": entry.filename, "bytes": entry.file_size,
                      "compressed_bytes": entry.compress_size} for entry in archive.infolist()]
        (args.root / "archive_inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
        member = next(entry for entry in archive.infolist()
                      if Path(entry.filename).name == "item_daily_features.csv")
        output = args.root / "item_daily_features.csv"
        with archive.open(member) as source, output.open("wb") as target:
            shutil.copyfileobj(source, target, 8 * 1024 * 1024)
    print(json.dumps({"extracted": str(output), "bytes": output.stat().st_size,
                      "archive_file_count": len(inventory)}), flush=True)


if __name__ == "__main__":
    main()
