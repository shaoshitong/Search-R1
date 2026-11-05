"""Capture exit/log for the separately frozen YouTube pilot."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

p=argparse.ArgumentParser();p.add_argument("research_root",type=Path);args=p.parse_args()
root=args.research_root/"v7/openchannelstats_empirical";code=root/"code";audit=root/"audit";audit.mkdir(parents=True,exist_ok=True)
attempt=len(list(audit.glob("run_*_exit.json")))+1;prefix="run_%02d"%attempt
command=["nice","-n","10","/home/shaoshitong/anaconda3/bin/python","-u",str(code/"analyze_openchannelstats.py"),"--research-root",str(args.research_root),"--protocol",str(code/"openchannelstats_protocol.json"),"--output",str(root/"results")]
env=os.environ.copy()
for key in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"]:env[key]="1"
started=datetime.now(timezone.utc).isoformat()
with (audit/(prefix+".log")).open("w") as log:
    proc=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env=env)
    for line in proc.stdout:log.write(line);log.flush();print(line,end="",flush=True)
    status=proc.wait()
record={"attempt":attempt,"started_utc":started,"finished_utc":datetime.now(timezone.utc).isoformat(),"actual_process_exit_code":status,"code_sha256":hashlib.sha256((code/"analyze_openchannelstats.py").read_bytes()).hexdigest(),"protocol_sha256":hashlib.sha256((code/"openchannelstats_protocol.json").read_bytes()).hexdigest(),"command":command}
(audit/(prefix+"_exit.json")).write_text(json.dumps(record,indent=2)+"\n");print(json.dumps({"run_exit":record}),flush=True)
raise SystemExit(status)
