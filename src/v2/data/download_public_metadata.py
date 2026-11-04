#!/usr/bin/env python3
"""Download pinned, explicitly non-gated metadata; store everything remotely."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

SOURCES = {
    "civitai_2_5m": {
        "repo": "hanruijiang/civitai-stable-diffusion-2.5m",
        "commit": "d0bbe735e7092473328196adb677534056997879",
        "file": "civitai-2023-11-14-sorted.jsonl.gz",
        "bytes": 767369614,
        "sha256": "95e2a43d91068e65bec42ed4e5d4bfa4aca0ec734bda6b81821021ed8f7478fe",
        "license": "Apache-2.0 (dataset card)", "gated": False,
    },
    "diffusiondb_2m": {
        "repo": "poloclub/diffusiondb",
        "commit": "fb620fbe49fa4420e0734bd9c0df11f51176b61f",
        "file": "metadata.parquet",
        "bytes": 194548652,
        "sha256": "eecd341187bc91c07f5994ad0660d40228ea025616fd57a509bef8323677c68f",
        "license": "CC0-1.0", "gated": False,
    },
    "civitai_model_trends_2026": {
        "repo": "Shiba-2-shiba/civitai-model-version-trends",
        "commit": "fb10c1c476d8201c496650d77c0aa33d46087248",
        "file": "snapshots/civitai.sqlite.gz",
        "bytes": 234590860,
        "sha256": "efc3181f03809d776a0cdbb3fc187e2288c23a48b5e76efb8689126bf4130501",
        "license": "not declared", "gated": False,
    },
}


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", choices=SOURCES)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    source = SOURCES[args.source]
    folder = args.root / args.source
    folder.mkdir(parents=True, exist_ok=True)
    filename = Path(source["file"]).name
    partial = folder / (filename + ".partial")
    final = folder / filename
    url = f"https://hf-mirror.com/datasets/{source['repo']}/resolve/{source['commit']}/{source['file']}?download=true"
    observations = []
    started = time.monotonic()
    with (folder / "download.stderr").open("w") as stderr:
        process = subprocess.Popen(
            ["curl", "-4", "-fL", "--connect-timeout", "8", "--max-time", "1200",
             "--speed-limit", "1024", "--speed-time", "60", "--retry", "2", "--retry-delay", "30", "--silent", "--show-error",
             "--output", str(partial), url], stderr=stderr,
        )
        while process.poll() is None:
            point = {"seconds": round(time.monotonic() - started, 2),
                     "bytes": partial.stat().st_size if partial.exists() else 0}
            observations.append(point)
            print(json.dumps({"source": args.source, "progress": point}), flush=True)
            time.sleep(5)
        exit_code = process.wait()
    actual_bytes = partial.stat().st_size if partial.exists() else 0
    actual_sha = digest_file(partial) if partial.exists() else None
    verified = exit_code == 0 and actual_bytes == source["bytes"] and actual_sha == source["sha256"]
    if verified:
        partial.rename(final)
    receipt = dict(source, source_key=args.source, url=url, curl_exit_code=exit_code,
                   actual_bytes=actual_bytes, actual_sha256=actual_sha, verified=verified,
                   path=str(final if verified else partial), progress=observations,
                   completed_utc=datetime.now(timezone.utc).isoformat(), images_downloaded=False)
    (folder / "download_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"download_result": receipt}), flush=True)
    if not verified:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
