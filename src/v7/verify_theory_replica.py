"""Client-side independent SHA audit of the two HKUST theory copies."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess

code = ('import hashlib,json,pathlib,sys; r=pathlib.Path(sys.argv[1]); '
        'print(json.dumps({str(p.relative_to(r)):{"bytes":p.stat().st_size,'
        '"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} '
        'for p in sorted(r.rglob("*")) if p.is_file()}))')
roots = {
    "hkust3": "/data/shaoshitong/ai_creator_game_20261104/v7/theory",
    "hkust2": "/data/shaoshitong/ai_creator_game_20261104/v7/theory_replica/theory",
}
results = {}
for host, root in roots.items():
    remote = shlex.join(["/home/shaoshitong/anaconda3/bin/python", "-c", code, root])
    completed = subprocess.run(["ssh", "-oBatchMode=yes", "-oConnectTimeout=8", host, remote],
                               capture_output=True, text=True, timeout=60, check=True)
    results[host] = json.loads(completed.stdout)
record = dict(verified_utc=datetime.now(timezone.utc).isoformat(), roots=roots,
              matching_files=len(results["hkust3"]),
              all_files_match=results["hkust3"] == results["hkust2"],
              total_bytes=sum(f["bytes"] for f in results["hkust3"].values()),
              manifest=results["hkust3"],
              storage_note="Both paths are ext4: an independent same-cluster replica, not shared NFS.")
target = Path(__file__).resolve().parents[2] / "docs/V7_AUDIT/theory_replica_verification.json"
target.write_text(json.dumps(record, indent=2) + "\n")
print({k:v for k,v in record.items() if k != "manifest"})
assert record["all_files_match"]
