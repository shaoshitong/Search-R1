"""Mac memory-only A800 -> existing HPC3 SSH-master result mirror.

No weights/raw records are written to this host. Only per-file checksums and
compact progress receipts are local. Uses the user's existing routes/auth.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import time

CONTROL='/Users/shaoshitong/Documents/remote_codex_control/control.py'
CONFIG='/Users/shaoshitong/Documents/remote_codex_control/access/hpc3-20261008/ssh_config'
SOCKET='/tmp/grounding-eval-20261019-501/hpc3-cm.sock'
SRC='/data/shaoshitong/grounding-eval-20261019'
DST='/data/user/jxu957/experiments/grounding-eval-strict-20261019/a800-mirror'
AUDIT='/Users/shaoshitong/Documents/Codex/2026-10-03/xia/outputs/grounding-eval-20261019'

def hpc3(command):
    return ['/usr/bin/ssh','-F',CONFIG,'-T','-S',SOCKET,'-oUser=mou176','-oBatchMode=yes','hpc3',command]

def main():
    ap=argparse.ArgumentParser(__doc__);ap.add_argument('--watch',action='store_true');args=ap.parse_args()
    spec=importlib.util.spec_from_file_location('grounding_user_control',CONTROL)
    ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl)
    ledger=Path(AUDIT)/'a800-mirror-ledger.json'
    prior=json.loads(ledger.read_text()) if ledger.exists() else {}
    while True:
        try:
            inventory=r'''
import pathlib,json,hashlib,time,os
root=pathlib.Path(ROOT);out={}
if root.exists():
 for parent,dirs,names in os.walk(root,followlinks=False):
  # Prune BEFORE traversal/hash: a corpus cache can contain a 14GB .jsonl
  # hardlink, which is neither an evaluation artifact nor a small inventory.
  dirs[:]=[d for d in dirs if not d.startswith(('cache','data-cache','hf-cache','arrow-cache','ray-diagnostic','.cache')) and not (pathlib.Path(parent)/d).is_symlink()]
  for name in names:
   p=pathlib.Path(parent)/name;rel=p.relative_to(root)
   if not p.is_file() or p.is_symlink():continue
   if p.suffix not in ('.json','.jsonl','.py','.md','.log') or '.partial' in p.name or p.name.endswith('.tmp'):continue
   # Closed/stable logs join the mirror; never race an actively growing log.
   if p.suffix=='.log' and time.time()-p.stat().st_mtime<60:continue
   h=hashlib.sha256(p.read_bytes()).hexdigest();out[str(rel)]={'bytes':p.stat().st_size,'sha256':h}
print(json.dumps(out))
'''.replace('ROOT',repr(SRC))
            got=subprocess.run(ctl.route('baai-a800','python3 -c '+shlex.quote(inventory)),capture_output=True,text=True,timeout=55)
            if got.returncode:raise RuntimeError('Source inventory SSH failed: '+got.stderr[-300:])
            current=json.loads(got.stdout);changed={p:v for p,v in current.items() if prior.get(p)!=v}
            backlog=len(changed)
            changed=dict(list(changed.items())[:256])
            if changed:
                source_code=('import tarfile,sys,pathlib;root=pathlib.Path('+repr(SRC)+');'
                    't=tarfile.open(fileobj=sys.stdout.buffer,mode="w|gz");'
                    '[t.add(root/p,arcname=p,recursive=False) for p in '+repr(list(changed))+'];t.close()')
                # Destination validates path, size and SHA before atomic publication.
                dest_code=r'''
import tarfile,sys,pathlib,hashlib,json,os
root=pathlib.Path(ROOT);root.mkdir(parents=True,exist_ok=True);expected=EXPECTED
seen=[]
with tarfile.open(fileobj=sys.stdin.buffer,mode='r|gz') as t:
 for member in t:
  name=member.name
  if name not in expected or not member.isfile() or '..' in pathlib.PurePosixPath(name).parts or name.startswith('/'):raise ValueError('Unexpected archive path')
  p=root/name;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.mirror-partial');h=hashlib.sha256();n=0
  with t.extractfile(member) as src,open(tmp,'wb') as dst:
   for b in iter(lambda:src.read(1048576),b''):dst.write(b);h.update(b);n+=len(b)
   dst.flush();os.fsync(dst.fileno())
  if {'bytes':n,'sha256':h.hexdigest()}!=expected[name]:raise ValueError('Source changed or SHA mismatch: '+name)
  os.replace(tmp,p)
  if hashlib.sha256(p.read_bytes()).hexdigest()!=expected[name]['sha256']:raise ValueError('Destination readback mismatch')
  seen.append(name)
if set(seen)!=set(expected):raise ValueError('Incomplete archive')
fd=os.open(root,os.O_RDONLY);os.fsync(fd);os.close(fd)
print(json.dumps({'verified_files':len(seen),'verified_bytes':sum(x['bytes'] for x in expected.values()),'root':str(root)}))
'''.replace('ROOT',repr(DST)).replace('EXPECTED',repr(changed))
                source=subprocess.Popen(ctl.route('baai-a800','python3 -c '+shlex.quote(source_code)),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                dest=subprocess.Popen(hpc3('python3 -c '+shlex.quote(dest_code)),stdin=source.stdout,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                source.stdout.close();stdout,stderr=dest.communicate(timeout=180);source.wait(timeout=10)
                if dest.returncode or source.returncode:raise RuntimeError('Mirror stream failed: '+stderr.decode()[-300:]+source.stderr.read().decode()[-300:])
                receipt=json.loads(stdout);prior.update(changed)
                ledger.write_text(json.dumps(prior,sort_keys=True))
            else:receipt=dict(verified_files=0,verified_bytes=0,root=DST)
            status=dict(utc_epoch=time.time(),ok=True,**receipt,total_files=len(prior),total_bytes=sum(x['bytes'] for x in prior.values()))
            (Path(AUDIT)/'a800-mirror-status.json').write_text(json.dumps(status,indent=2));print(json.dumps(status),flush=True)
        except Exception as error:
            status=dict(utc_epoch=time.time(),ok=False,error=type(error).__name__+':'+str(error))
            (Path(AUDIT)/'a800-mirror-status.json').write_text(json.dumps(status));print(json.dumps(status),flush=True)
            if not args.watch:raise
        if not args.watch:return
        if 'backlog' not in locals() or backlog<=256:time.sleep(30)

if __name__=='__main__':main()
