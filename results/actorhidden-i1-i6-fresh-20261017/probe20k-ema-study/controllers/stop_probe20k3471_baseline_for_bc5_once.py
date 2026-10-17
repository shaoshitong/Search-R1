from pathlib import Path
import sys,json,datetime,hashlib
sys.path.insert(0,'/Users/shaoshitong/Documents/Search-G1/actorhidden-period20-20261016')
from reproduction.h1008_qprobe.established_controller_transport import remote
script=r'''
import pathlib,json,subprocess,time
b=pathlib.Path('/mnt/data/shaoshitong/search-g1-prefix-selection200-probe20k-fit-20261017-0bdddc20840b-r48-no-loop-checks-resume1');p=b/'preparation/human-stop-baseline-run-next-two-arms-once.json'
if p.exists():print(p.read_text());raise SystemExit
r=subprocess.run(['scontrol','show','job','3471','-o'],capture_output=True,text=True,timeout=20);assert r.returncode==0
native=dict(x.split('=',1)for x in r.stdout.split()if '='in x)
assert native['Command']==str(b/'preparation/native-probe20k-fit.sh') and native['JobState']=='RUNNING'
c=json.loads((b/'preparation/launch-config.json').read_text());assert c['source_commit']=='b5a5e79e528297e44391288258570ca8265829be'
latest={name:json.loads((b/'arms'/name/'latest.json').read_text())if(b/'arms'/name/'latest.json').exists()else None for name in ('data_ema','data_ema_group2','data_ema_mixup')}
assert latest['data_ema']['epoch']>=1
v=dict(human_instruction='那可以停掉了，这说明训练5epoch已经足以分析问题了，然后跑1+2+3和1+2+4',
 scope='Stop only baseline arm A in own3471. Preserve actual results. Run B groupweight2 and C hidden/soft-label mixup for five epochs each from original matched source head40 Adam200. No new rollout; formal actor remains stopped',
 explicit_human_stop=True,whole_study_paused=False,native_before=native,latest_before=latest,checked_unix=time.time())
with p.open('x')as f:json.dump(v,f,indent=2)
r=subprocess.run(['scancel','3471'],capture_output=True,text=True,timeout=20)
v['scancel_returncode']=r.returncode;v['stderr']=r.stderr;assert r.returncode==0
(b/'preparation/human-stop-baseline-run-next-two-arms-result.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v))
'''
p=Path('work')/('probe20k3471-human-stop-baseline-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')
v=json.loads(remote(script,str(p.with_suffix('.transport.json'))));p.write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(dict(actual=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),result=v)))
