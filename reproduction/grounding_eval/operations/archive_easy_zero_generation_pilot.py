"""Preserve a proven exited, zero-generation Easy admission failure before retry."""
import importlib.util,json,shlex,subprocess
from pathlib import Path

REMOTE = r'''
import json,pathlib,os,fcntl,time,hashlib
base=pathlib.Path('/data/shaoshitong/grounding-eval-20261019')
src=base/'easy-selection-normal-pilot96'; dst=base/'easy-selection-normal-pilot96-admission-failed-dde6641b'
assert src.is_dir() and not dst.exists()
exit=json.loads((src/'control/exit.json').read_text()); launch=json.loads((src/'control/launch.json').read_text())
assert exit['exit_code']==1 and launch['source_commit']=='dde6641b'
for pid in [exit['pid'],exit['child_pid']]:
 assert not(pathlib.Path('/proc')/str(pid)).exists(),'Old task process remains; inspect before archive'
assert not(src/'raw').exists() and not(src/'normal-binding.json').exists() and not(src/'complete.json').exists()
locks=[]
for path in [src/'.normal.lock',src/'control/.controller.lock']:
 if path.exists():
  f=path.open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
files={str(p.relative_to(src)):hashlib.sha256(p.read_bytes()).hexdigest() for p in src.rglob('*') if p.is_file()}
os.rename(src,dst)
assert files=={str(p.relative_to(dst)):hashlib.sha256(p.read_bytes()).hexdigest() for p in dst.rglob('*') if p.is_file()}
receipt=dict(state='ZERO_GENERATION_ADMISSION_FAILURE_PRESERVED',source=str(src),archive=str(dst),source_commit=launch['source_commit'],exit_code=1,files_sha256=files,at=time.time(),raw_rows=0,model_loaded=False)
p=base/'easy-pilot-zero-generation-archive.json'
with p.open('x')as f:json.dump(receipt,f,indent=2);f.flush();os.fsync(f.fileno())
print(json.dumps(receipt))
'''
def main():
 spec=importlib.util.spec_from_file_location('control','/Users/shaoshitong/Documents/remote_codex_control/control.py')
 ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl)
 cmd=ctl.route('baai-a800','/usr/bin/python3 -c '+shlex.quote(REMOTE));cmd[1:1]=['-o','ServerAliveInterval=5','-o','ServerAliveCountMax=6']
 out=subprocess.run(cmd,text=True,capture_output=True,timeout=45)
 assert out.returncode==0,out.stderr[-1500:]
 receipt=json.loads(out.stdout.strip().splitlines()[-1])
 Path('/Users/shaoshitong/Documents/Codex/2026-10-03/xia/outputs/grounding-eval-20261019/easy-pilot-zero-generation-archive.json').write_text(json.dumps(receipt,indent=2))
 print(json.dumps(receipt))
if __name__=='__main__':main()
