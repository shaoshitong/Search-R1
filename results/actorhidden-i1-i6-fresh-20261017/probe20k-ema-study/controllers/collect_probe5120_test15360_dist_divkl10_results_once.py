from pathlib import Path
import sys,json,datetime,hashlib
sys.path.insert(0,'/Users/shaoshitong/Documents/Search-G1/actorhidden-period20-20261016')
from reproduction.h1008_qprobe.established_controller_transport import remote
inner=r'''
import pathlib,json,sys
b=pathlib.Path('/mnt/data/shaoshitong/search-g1-prefix-selection200-probe-dist-divkl-resplit5120-15360-20261018-0bdddc20840b-eval15360fix');sys.path.insert(0,str(b/'repo'))
from reproduction.prefix_selection200.probe20k_analysis import aggregate
p=b/'analysis/ten-epoch-5120-test15360-dist-divkl-aggregate.json';p.parent.mkdir(exist_ok=True)
if p.exists():print(p.read_text());raise SystemExit
assert (b/'logs/pipeline-complete.json').exists()
v=aggregate(b)
with p.open('x')as f:json.dump(v,f,indent=2)
print(json.dumps(v))
'''
script='import os,subprocess\nr=subprocess.run(["/dataset/shaoshitong/Search-G1-common8/env/bin/python","-c",'+repr(inner)+'],env=dict(os.environ,CUDA_VISIBLE_DEVICES=""),text=True,capture_output=True,timeout=180);assert r.returncode==0,r.stderr;print(r.stdout)'
p=Path('work/probe5120-test15360-dist-divkl10-final-aggregate.json');v=json.loads(remote(script,str(p.with_name(p.stem+'-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.transport.json'))));p.write_text(json.dumps(v,indent=2)+'\n')
print(json.dumps(dict(actual=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),arms=[dict(name=a['arm']['name'],epochs=[e['epoch']for e in a['epochs']],complete=bool(a['complete']))for a in v['arms']])))
