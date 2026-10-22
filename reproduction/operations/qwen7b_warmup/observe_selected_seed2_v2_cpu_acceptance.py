"""Read-only observer of the corrected-v2 serial terminal CP acceptance coordinator."""
import argparse,hashlib,importlib.util,json,os,shlex,subprocess,time
from pathlib import Path
R=Path(__file__).resolve().parents[3];CONTROL=R.parents[1]
REMOTE='''from pathlib import Path
import hashlib,json,time
A=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021');C=A/'selected-seed2-v2-checkpoint-CPU-coordinator-20261023-v1';O=A/'seed2-matched-window-terminal-CPU-acceptance-20261023-v1';out=dict(time=time.time(),receipts={})
for suite,cell,e in [('online-packed-current64-seed19137-plan2026102302-L15P80-8192-20261023-v1','current64',5),('online-packed-matched-MSE-M5-seed19137-plan2026102302-L15P80-8192-20261023-v1','mse-M5',7),('online-packed-window2018-seed19137-plan2026102302-L15P80-8192-20261023-v2','window2018',10)]:
 p=(A/'seed2-window2018-v2-terminal-CPU-acceptance-20261023-v1'if cell=='window2018'else O)/suite/cell/('epoch-%02d'%e)/'accepted.json'
 if p.is_file():
  raw=p.read_bytes();out['receipts'][cell]=dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),accepted=json.loads(raw))
for name in('controller.json','progress.json','results-so-far.json','complete.json','failure.json'):
 if(C/name).is_file():out[name]=json.loads((C/name).read_bytes())
ctrl=out.get('controller.json');stat=Path('/proc/%d/stat'%ctrl['pid'])if ctrl else None;out['coordinator_identity_live']=bool(stat and stat.exists()and stat.read_text().rsplit(')',1)[1].split()[19]==ctrl['start_ticks']);print(json.dumps(out))
'''
def atomic(p,v):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(v,sort_keys=True)+'\n');q.replace(p)
def main():
 p=argparse.ArgumentParser();p.add_argument('--interval',type=int,default=60);a=p.parse_args();assert a.interval>=30
 out=R/'reproduction/reports/qwen7b-algorithm-improvement-20261021/seed2-window2018-v2-CPU-acceptance';out.mkdir(parents=True,exist_ok=True)
 spec=importlib.util.spec_from_file_location('private_control',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl);command='python3 -c '+shlex.quote(REMOTE);last=None
 atomic(out/'observer-controller.json',dict(pid=os.getpid(),time=time.time(),role='read_only_metadata_no_CP_loader'))
 while True:
  try:
   address=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip();response=ctl.run(command,account='mou176',source_address=address,timeout=35)
   if response.returncode:raise RuntimeError('Read-only control unavailable; output omitted')
   result=json.loads(response.stdout);atomic(out/'observer-latest.json',result);key=json.dumps(dict(receipts=list(result['receipts']),progress=result.get('progress.json',{}).get('cell'),stage=result.get('progress.json',{}).get('stage'),complete='complete.json'in result,failure='failure.json'in result,live=result['coordinator_identity_live']),sort_keys=True)
   with(out/'observer-observations.jsonl').open('a')as stream:stream.write(json.dumps(dict(time=result['time'],state=json.loads(key)))+'\n')
   if key!=last:print(json.dumps(dict(time=result['time'],state=json.loads(key))),flush=True);last=key
   if'complete.json'in result:atomic(out/'observer-complete.json',result);return
   if'failure.json'in result or not result['coordinator_identity_live']:raise RuntimeError('Durable coordinator exited or reported failure before all acceptances')
  except Exception as error:
   atomic(out/'observer-error.json',dict(time=time.time(),type=type(error).__name__,message=str(error)));raise
  time.sleep(a.interval)
if __name__=='__main__':main()
