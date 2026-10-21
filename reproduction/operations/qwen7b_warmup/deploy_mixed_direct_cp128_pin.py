"""Publish/run a bounded metadata-only CP128 hardlink watcher; never submits GPU."""
import argparse,base64,hashlib,importlib.util,json,shlex,subprocess,zlib
from pathlib import Path
R=Path(__file__).resolve().parents[3];CONTROL=R.parents[1];P='reproduction/operations/qwen7b_warmup/';REL=P+'mixed_direct_cp128_pin.py';BASE='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/mixed-direct-official-cp128-eval-v1';BRANCH='codex/mixed-direct-cp128-eval-20261022';OUT=R/'reproduction/reports/qwen7b-algorithm-improvement-20261021/mixed-direct-cp128-eval'
WATCH='''import fcntl,json,os,sys,time,traceback
from pathlib import Path
b=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(Path(__file__).parent))
from mixed_direct_cp128_pin import pin
with(b/'pin-watch.lock').open('a+')as f:
 fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for attempt in range(61):
  try:
   result=pin();result['watch_attempt']=attempt;result['watch_epoch']=time.time()
   raw=(json.dumps(result,sort_keys=True)+chr(10)).encode();temporary=b/'pin-watch-status.partial';temporary.write_bytes(raw);os.replace(str(temporary),str(b/'pin-watch-status.json'));print(raw.decode(),flush=True)
   if result['status'].startswith(('MAIN_CP128_','CP128_IMMUTABLE_')):break
  except Exception as e:
   error=dict(status='CP128_METADATA_PIN_WATCH_FAILED',error_type=type(e).__name__,error=str(e),epoch=time.time());(b/'pin-watch-failure.json').write_text(json.dumps(error,sort_keys=True)+chr(10));traceback.print_exc();raise
  if attempt==60:raise RuntimeError('Finite CP128 metadata watcher expired')
  time.sleep(30)
'''
def git(c,p):return subprocess.check_output(['git','show',c+':'+p],cwd=R)
def main():
 p=argparse.ArgumentParser();p.add_argument('--action',choices=('start','status','pin'),required=True);a=p.parse_args();commit=subprocess.check_output(['git','log','-1','--format=%H','refs/heads/'+BRANCH,'--',REL],cwd=R).decode().strip();data=git(commit,REL);assert data==(R/REL).read_bytes();remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+BRANCH],cwd=R).decode().split()[0];subprocess.check_call(['git','merge-base','--is-ancestor',commit,remote],cwd=R)
 spec=importlib.util.spec_from_file_location('private_cp128_control',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl);ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0']).decode().strip()
 common="import pathlib,json,hashlib,os,subprocess,time,fcntl,base64\nb=pathlib.Path("+repr(BASE)+");b.mkdir(parents=True,exist_ok=True);code=b/"+repr('pin-code-'+commit[:12])+";code.mkdir(exist_ok=True)\ndef proof(p):\n raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())\ndef once(p,data):\n if p.exists():assert p.read_bytes()==data\n else:p.write_bytes(data);p.chmod(0o444)\n"
 if a.action=='start':
  body="once(code/'mixed_direct_cp128_pin.py',base64.b64decode("+repr(base64.b64encode(data).decode())+"));once(code/'watch.py',base64.b64decode("+repr(base64.b64encode(WATCH.encode()).decode())+"));publication=dict(commit="+repr(commit)+",module=proof(code/'mixed_direct_cp128_pin.py'),watcher=proof(code/'watch.py'),checkpoint_payload_reads=False,GPU_submitted=False);once(b/'pin-source-publication.json',(json.dumps(publication,sort_keys=True)+chr(10)).encode());guard=(b/'pin-start.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)\ntry:\n probe=(b/'pin-watch.lock').open('a+');fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)\nexcept BlockingIOError:\n print(json.dumps(dict(status='CP128_METADATA_WATCH_ALREADY_ACTIVE',publication=proof(b/'pin-source-publication.json'))));raise SystemExit(0)\nif(b/'snapshot-pin.json').exists():print(json.dumps(dict(status='CP128_ALREADY_PINNED',pin=proof(b/'snapshot-pin.json'))));raise SystemExit(0)\nassert not(b/'pin-watch-failure.json').exists(),'Preserve/reconcile prior watcher failure';probe.close();log=(b/'pin-watch.log').open('ab');child=subprocess.Popen(['python3','-u',str(code/'watch.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True);print(json.dumps(dict(status='FINITE_CP128_METADATA_WATCH_STARTED',pid=child.pid,publication=proof(b/'pin-source-publication.json'),epoch=time.time(),GPU_submitted=False)))"
 elif a.action=='pin':body="import sys;sys.path.insert(0,str(code));from mixed_direct_cp128_pin import pin;print(json.dumps(pin()))"
 else:body="out=dict(epoch=time.time());\nfor name in('pin-source-publication.json','pin-watch-status.json','snapshot-pin.json','pin-watch-failure.json'):\n p=b/name\n if p.exists():out[name[:-5]]=json.loads(p.read_bytes())\nprint(json.dumps(out))"
 source=common+body;compile(source,'cp128_metadata_control','exec');packed=base64.b64encode(zlib.compress(source.encode())).decode();command='python3 -c '+shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(packed)+')))');assert len(command.encode())<16000;result=ctl.run(command,account='mou176',source_address=ip,timeout=45)
 if result.returncode:raise RuntimeError(result.stdout[-3000:]+result.stderr[-2000:])
 result=json.loads(result.stdout.splitlines()[-1]);OUT.mkdir(parents=True,exist_ok=True);(OUT/('actual-pin-'+a.action+'.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
