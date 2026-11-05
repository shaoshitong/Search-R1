"""Hash only this task's artifacts and explicit public/source CSV inputs."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument("base",type=Path);p.add_argument("--replica",action="store_true");args=p.parse_args()
files={}
def record(key,path):files[key]={"bytes":path.stat().st_size,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
for folder in ["empirical","openchannelstats_empirical"]:
    for path in sorted((args.base/folder).rglob("*")):
        if path.is_file():record(str(path.relative_to(args.base)),path)
source=args.base/"source_data/item_daily_features.csv" if args.replica else args.base.parent/"v3/kuairec/item_daily_features.csv"
record("source_data/item_daily_features.csv",source)
summary=json.loads((args.base/"openchannelstats_empirical/results/summary.json").read_text())
for original in summary["input_files"]:
    relative="source_discovery/"+original.split("/v7/source_discovery/",1)[1]
    record(relative,args.base/relative)
print(json.dumps({"checked_utc":datetime.now(timezone.utc).isoformat(),"base":str(args.base),"replica":args.replica,"files":files},indent=2))
