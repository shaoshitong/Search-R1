"""Obtain the official metadata/log-only KuaiRand-Pure archive on HKUST."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import time

URL = "https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz"
EXPECTED_MD5 = "0820331067a3784d9691136f772b35a7"
EXPECTED_BYTES = 47432272


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--no-extract", action="store_true")
    args = parser.parse_args()
    raw, audit = args.root / "raw", args.root / "audit"
    raw.mkdir(parents=True, exist_ok=True)
    audit.mkdir(parents=True, exist_ok=True)
    target = raw / "KuaiRand-Pure.tar.gz"
    points = []
    start = time.monotonic()
    with (audit / "download.stderr").open("w") as err:
        process = subprocess.Popen(["curl", "-4", "-fL", "--silent", "--show-error",
                                    "--connect-timeout", "8", "--max-time", "600",
                                    "--speed-time", "60", "--speed-limit", "1024",
                                    "-o", str(target), URL], stderr=err)
        while process.poll() is None:
            point = {"elapsed_seconds": round(time.monotonic()-start, 2),
                     "bytes": target.stat().st_size if target.exists() else 0}
            points.append(point)
            print(json.dumps({"download_progress":point}), flush=True)
            time.sleep(1)
    data = target.read_bytes() if target.exists() else b""
    md5, sha = hashlib.md5(data).hexdigest(), hashlib.sha256(data).hexdigest()
    verified = process.returncode == 0 and len(data) == EXPECTED_BYTES and md5 == EXPECTED_MD5
    receipt = {"source_url": URL, "source_record":"https://zenodo.org/records/10439422",
               "license":"CC-BY-SA-4.0 as declared by the official project repository",
               "official_repository":"https://github.com/chongminggao/KuaiRand",
               "expected_bytes": EXPECTED_BYTES, "bytes":len(data),
               "expected_md5":EXPECTED_MD5,"md5":md5,"sha256":sha,
               "curl_exit_code":process.returncode,"verified":verified,"progress":points,
               "completed_utc":datetime.now(timezone.utc).isoformat()}
    (audit/"download_receipt.json").write_text(json.dumps(receipt, indent=2))
    if not verified:
        raise SystemExit("Official archive download or checksum did not complete")
    if args.no_extract:
        print(json.dumps({"complete":True,"bytes":len(data),"extracted":False}),flush=True)
        return
    with tarfile.open(target) as archive:
        members = archive.getmembers()
        for member in members:
            name = Path(member.name)
            if name.is_absolute() or ".." in name.parts or member.issym() or member.islnk():
                raise ValueError("Unexpected archive path")
        archive.extractall(raw)
    print(json.dumps({"complete":True,"bytes":len(data),"files":[m.name for m in members]}),flush=True)


if __name__ == "__main__":
    main()
