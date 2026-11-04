#!/usr/bin/env python3
"""One anonymous robots/API probe; no retries, proxy changes, cookies or media."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess


def get(url):
    started = datetime.now(timezone.utc).isoformat()
    response = subprocess.run([
        "curl", "-q", "-4", "-L", "--connect-timeout", "8", "--max-time", "20",
        "--silent", "--show-error", "--user-agent", "AI-Creator-Game-Research/0.1 metadata-only",
        "--write-out", "\n%{http_code} %{size_download} %{time_total}", url,
    ], capture_output=True, text=True)
    body, _, metrics = response.stdout.rpartition("\n")
    result = {"url": url, "started_utc": started,
              "completed_utc": datetime.now(timezone.utc).isoformat(),
              "curl_exit": response.returncode, "metrics": metrics, "stderr": response.stderr}
    return result, body


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    robots, text = get("https://civitai.com/robots.txt")
    if robots["curl_exit"] == 0 and robots["metrics"].startswith("200 "):
        (args.output_root / "robots.txt").write_text(text)
    # This is deliberately a probe, not a bulk crawler. Review fetched robots
    # and public API documentation before enabling any further collection.
    (args.output_root / "robots_probe.json").write_text(json.dumps(robots, indent=2) + "\n")
    print(json.dumps(robots))


if __name__ == "__main__":
    main()
