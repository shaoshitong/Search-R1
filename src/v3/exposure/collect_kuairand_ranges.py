"""Resume the public official archive in eight bounded HTTP Range streams."""
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import time
import urllib.request

URL = "https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz"
SIZE = 47432272
MD5 = "0820331067a3784d9691136f772b35a7"


def main():
    root = Path(sys.argv[1])
    parts = root / "raw" / "official_range_parts"
    parts.mkdir(parents=True, exist_ok=True)
    audit = root / "audit"
    audit.mkdir(exist_ok=True)
    block = 1024 * 1024
    jobs = [(i, start, min(start+block, SIZE)-1) for i, start in enumerate(range(0, SIZE, block))]
    started = time.monotonic()
    events = []

    def fetch(job):
        i, start, end = job
        path = parts / f"part-{i:03d}"
        if path.exists() and path.stat().st_size == end-start+1:
            return {"part": i, "bytes": path.stat().st_size, "resumed": True}
        existing = path.stat().st_size if path.exists() else 0
        request_start = start + existing
        request = urllib.request.Request(URL, headers={"Range": f"bytes={request_start}-{end}"})
        with urllib.request.urlopen(request, timeout=60) as response:
            expected_range = f"bytes {request_start}-{end}/{SIZE}"
            if response.status != 206 or response.headers.get("Content-Range") != expected_range:
                raise ValueError(f"Range response mismatch: {response.status}, {response.headers.get('Content-Range')}")
            with path.open("ab") as stream:
                while chunk := response.read(65536):
                    stream.write(chunk)
        if path.stat().st_size != end-start+1:
            raise ValueError(f"Incomplete public archive part {i}")
        return {"part": i, "bytes": path.stat().st_size, "resumed": bool(existing), "prefix_bytes":existing}

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for event in pool.map(fetch, jobs):
            event["elapsed_seconds"] = round(time.monotonic()-started, 2)
            events.append(event)
            print(json.dumps(event), flush=True)
    target = root / "raw" / "KuaiRand-Pure.verified.tar.gz"
    md5, sha = hashlib.md5(), hashlib.sha256()
    with target.open("wb") as output:
        for i, _, _ in jobs:
            chunk = (parts / f"part-{i:03d}").read_bytes()
            output.write(chunk)
            md5.update(chunk)
            sha.update(chunk)
    verified = target.stat().st_size == SIZE and md5.hexdigest() == MD5
    receipt = {"source_url": URL, "expected_bytes": SIZE, "bytes": target.stat().st_size,
               "expected_md5": MD5, "md5": md5.hexdigest(), "sha256": sha.hexdigest(),
               "verified": verified, "range_streams": 8, "part_events": events,
               "completed_utc": datetime.now(timezone.utc).isoformat()}
    (audit / "official_range_receipt.json").write_text(json.dumps(receipt, indent=2))
    if not verified:
        raise ValueError("Official archive digest mismatch")
    with tarfile.open(target) as archive:
        archive.extractall(root / "raw", filter="data")
    print(json.dumps({"complete": True, "sha256": sha.hexdigest()}), flush=True)


if __name__ == "__main__":
    main()
