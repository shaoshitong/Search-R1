"""Run the installed local DeepSeek Harness directly; preserve actual completion."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

repo = Path(__file__).resolve().parents[2]
prompt = Path(sys.argv[1]).resolve()
prefix = repo / "docs/V8_AUDIT" / sys.argv[2]
prefix.parent.mkdir(parents=True, exist_ok=True)
command = ["/Applications/DeepSeek Harness.app/Contents/Resources/runtime/cli/bin/dsh",
           "--profile", "headless",
           "--patch", str(Path(__file__).with_name("dpsk_account_patch.json")),
           "--json"]
if len(sys.argv) > 3:
    command.extend(["--session-id", sys.argv[3]])
command.append("-")
started = datetime.now(timezone.utc).isoformat()
with prompt.open("rb") as source, prefix.with_suffix(".ndjson").open("wb") as stdout, \
        prefix.with_suffix(".stderr").open("wb") as stderr:
    process = subprocess.Popen(command, cwd=repo, stdin=source, stdout=stdout, stderr=stderr)
    status = process.wait()
record = dict(started_utc=started, finished_utc=datetime.now(timezone.utc).isoformat(),
              command=command, cwd=str(repo), prompt=str(prompt),
              prompt_sha256=hashlib.sha256(prompt.read_bytes()).hexdigest(),
              actual_exit_code=status,
              event_bytes=prefix.with_suffix(".ndjson").stat().st_size,
              stderr_bytes=prefix.with_suffix(".stderr").stat().st_size)
prefix.with_name(prefix.name + "_receipt.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record), flush=True)
sys.exit(status)
