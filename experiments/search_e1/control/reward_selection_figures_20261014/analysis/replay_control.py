from pathlib import Path
import sys,json,base64,shlex
from h100_remote import call
D=Path(__file__).resolve().parent;remote='/mnt/data/shaoshitong/reward-std-replay-20261014-2330'
if sys.argv[-1]=='launch':
 root=D.parent/'g1-reward-correlation-20261010';files={'worker.py':(D/'replay_mixed200.py').read_bytes(),'C_stream.py':(root/'reconstruct_train_raw_c_preflight_v2.py').read_bytes(),'replay_core.py':(root/'historical_replay/replay_core.py').read_bytes(),'stats.py':(D.parent/'g1-diversity-20261014/aggregate_remote.py').read_text().split('MODE=MODE_VALUE')[0].encode()};payload={n:base64.b64encode(b).decode() for n,b in files.items()}
 code='''from pathlib import Path
import json,base64,subprocess,os
r=Path(REMOTE);r.mkdir(mode=0o700,exist_ok=False)
for n,b in FILES.items():(r/n).write_bytes(base64.b64decode(b))
p=subprocess.Popen(['/mnt/data/shaoshitong/search-g1-speed-resume5-0553ade-20261012/env/bin/python','-B',str(r/'worker.py')],cwd=str(r),stdout=(r/'stdout.log').open('w'),stderr=(r/'stderr.log').open('w'),start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'));(r/'launch.json').write_text(json.dumps({'pid':p.pid,'script':str(r/'worker.py')}));print(json.dumps({'pid':p.pid,'remote':str(r)}))'''.replace('REMOTE',repr(remote)).replace('FILES',repr(payload))
else:
 code='''from pathlib import Path
import json,subprocess
r=Path(REMOTE);v=json.load((r/'launch.json').open());o={'pid':v['pid'],'process':subprocess.run(['ps','-p',str(v['pid']),'-o','pid=,etime=,stat=,comm='],text=True,capture_output=True).stdout}
if (r/'status.json').exists():o['status']=json.load((r/'status.json').open())
if (r/'aggregate.json').exists():o['aggregate']=json.load((r/'aggregate.json').open())
if o.get('status',{}).get('stage')=='FAILED':o['stderr']=(r/'stderr.log').read_text()[-3000:]
print(json.dumps(o))'''.replace('REMOTE',repr(remote))
v=call('h100-8',code,100);(D/('replay-'+sys.argv[-1]+'.json')).write_text(json.dumps(v,indent=2))
if 'aggregate' in v:(D/'mixed200-soft-verified.json').write_text(json.dumps(v.pop('aggregate'),ensure_ascii=False,indent=2))
print(json.dumps(v))
