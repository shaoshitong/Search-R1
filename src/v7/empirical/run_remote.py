"""Execute CPU-only analysis and preserve actual exit code for every attempt."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

parser=argparse.ArgumentParser();parser.add_argument("research_root",type=Path);args=parser.parse_args()
root=args.research_root/"v7/empirical";code=root/"code";audit=root/"audit";audit.mkdir(parents=True,exist_ok=True)
number=len(list(audit.glob("run_*_exit.json")))+1;prefix="run_%02d"%number
command=["nice","-n","10","/home/shaoshitong/anaconda3/bin/python","-u",str(code/"analyze_states.py"),"--source",str(args.research_root/"v3/kuairec/item_daily_features.csv"),"--output",str(root/"results"),"--protocol",str(code/"protocol.json")]
env=os.environ.copy()
for name in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"]:env[name]="1"
started=datetime.now(timezone.utc).isoformat()
with (audit/(prefix+".log")).open("w") as log:
    process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env=env)
    for line in process.stdout:log.write(line);log.flush();print(line,end="",flush=True)
    status=process.wait()
receipt={"attempt":number,"started_utc":started,"finished_utc":datetime.now(timezone.utc).isoformat(),"actual_process_exit_code":status,"command":command,
         "code_sha256":hashlib.sha256((code/"analyze_states.py").read_bytes()).hexdigest(),"protocol_sha256":hashlib.sha256((code/"protocol.json").read_bytes()).hexdigest(),
         "thread_environment":{k:env[k] for k in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"]}}
(audit/(prefix+"_exit.json")).write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps({"run_exit":receipt}),flush=True)
raise SystemExit(status)
