"""读取 HKUST4 服务、四个 rank 的训练进度和 GPU 进程；不改变任何任务。"""
import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
meta = json.loads((ROOT / 'deployment-hkust4.json').read_text())
probe = r'''
import pathlib,json,subprocess,datetime
meta=META
root=pathlib.Path(meta['output'])
def read(args):
 p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
 return dict(exit=p.returncode,out=p.stdout,err=p.stderr)
progress=[json.loads(p.read_text()) for p in sorted((root/'progress').glob('rank-[0-9].json'))]
log=root/'train.log'
tail=''
if log.exists():
 with log.open('rb') as f:
  f.seek(max(0,log.stat().st_size-7000));tail=f.read().decode(errors='replace')
r=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),progress=progress,
 service=read(['systemctl','show',meta['unit']+'.service','-p','ActiveState','-p','SubState','-p','MainPID',
 '-p','Result','-p','ExecMainStatus','-p','ActiveEnterTimestamp','-p','RuntimeMaxUSec','-p','KillMode','-p','NRestarts','-p','User']),
 gpu=read(['nvidia-smi','-i','2,3,5,6','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits']),
 processes=read(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits']),log_tail=tail)
print('STATUS_JSON:'+json.dumps(r))
'''.replace('META', repr(meta))
p = argparse.ArgumentParser();p.add_argument('--save');a = p.parse_args()
r = subprocess.run(['python3','/Users/shaoshitong/Documents/remote_codex_control_cc/control.py','run','--timeout','25','lab_4090_4','--','python3','-c',probe],capture_output=True,text=True,timeout=35)
assert r.returncode == 0, r.stderr
s = json.loads(next(x.split(':',1)[1] for x in r.stdout.splitlines() if x.startswith('STATUS_JSON:')))
if a.save: Path(a.save).write_text(json.dumps(s,indent=2)+'\n')
print(json.dumps(s,indent=2))
