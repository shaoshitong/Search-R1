"""Capture the actual process exit and log of the independent supplement."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

parser=argparse.ArgumentParser()
parser.add_argument("research_root",type=Path)
args=parser.parse_args()
root=args.research_root/"v5/kuairec"
audit=root/"audit";audit.mkdir(parents=True,exist_ok=True)
attempt=len(list(audit.glob("new_upload_run_*_exit.json")))+1
prefix="new_upload_run_%02d"%attempt
code=root/"code"
command=["/home/shaoshitong/anaconda3/bin/python","-u",str(code/"analyze_new_uploads.py"),
         "--source",str(args.research_root/"v3/kuairec/item_daily_features.csv"),
         "--output",str(root/"new_upload_results"),"--protocol",str(code/"new_upload_protocol.json")]
started=datetime.now(timezone.utc).isoformat()
with (audit/(prefix+".log")).open("w") as log:
    process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    for line in process.stdout:
        log.write(line);log.flush();print(line,end="",flush=True)
    status=process.wait()
report={"attempt":attempt,"started_utc":started,"finished_utc":datetime.now(timezone.utc).isoformat(),
        "command":command,"actual_process_exit_code":status,
        "source_code_sha256":hashlib.sha256((code/"analyze_new_uploads.py").read_bytes()).hexdigest(),
        "helper_source_sha256":hashlib.sha256((code/"analyze_transfer.py").read_bytes()).hexdigest(),
        "protocol_sha256":hashlib.sha256((code/"new_upload_protocol.json").read_bytes()).hexdigest(),
        "log_path":str(audit/(prefix+".log"))}
(audit/(prefix+"_exit.json")).write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps({"run_exit":report}),flush=True)
raise SystemExit(status)
