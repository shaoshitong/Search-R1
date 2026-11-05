"""Capture actual subprocess completion without changing any earlier run."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(sys.argv[1])
root.mkdir(parents=True, exist_ok=True)
attempt = len(list(root.glob("run_*_exit.json"))) + 1
log = root / f"run_{attempt:02d}.log"
env = os.environ.copy()
env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
cmd = [sys.executable, "-u", str(Path(__file__).with_name("dynamic_game.py")), "--output", str(root)]
if len(sys.argv) > 2:
    cmd.extend(["--protocol", sys.argv[2]])
started = datetime.now(timezone.utc).isoformat()
with log.open("w") as stream:
    completed = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT, env=env)
receipt = dict(started_utc=started, finished_utc=datetime.now(timezone.utc).isoformat(),
               command=cmd, log=str(log), exit_code=completed.returncode)
(root / f"run_{attempt:02d}_exit.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt), flush=True)
sys.exit(completed.returncode)
