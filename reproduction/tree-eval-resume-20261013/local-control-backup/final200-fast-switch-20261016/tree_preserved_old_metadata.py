from pathlib import Path
import os,sys,shlex,json,time,hashlib
sys.path.insert(0,'work/tree-speed-repair-20261008');import ssh_retained18 as ssh
os.environ['TREE_MGMT_CONTROL_PATH']='/tmp/tree-mgmt-3fw9a1ab/control'
src=r'''import json,time,hashlib
from pathlib import Path
root=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo');run=root/'runs/tree-branpo-fresh200-3360'
def item(p,small=False):
 x=dict(path=str(p),exists=p.is_file())
 if x['exists']:
  s=p.stat();x.update(bytes=s.st_size,mtime_ns=s.st_mtime_ns)
  if small:
   assert s.st_size<262144;b=p.read_bytes();x.update(sha256=hashlib.sha256(b).hexdigest(),value=json.loads(b))
 return x
source=root/'instruct300/candidates/branpo-fresh200-8b6f99d1b3353145200eb3fa7c16a579a79a987c/manifest.json'
proofs=[source,run/'checkpoints/training_state/global_step_200/manifest.json',run/'output/eval-handoff-hf-acceptance/identity-bound-handoff-hf.json',run/'output/eval-handoff-hf-acceptance/independent-composite-acceptance.json']
x=dict(epoch=time.time(),scope='Read-only current existence/size and small receipt SHA; no old model/PT/raw-score hash, no mutation',run_exists=run.is_dir(),proofs=[item(p,True) for p in proofs],native200_files=[item(p) for p in sorted((run/'checkpoints/training_state/global_step_200').glob('*')) if p.is_file()],HF200_files=[item(p) for p in sorted((run/'checkpoints/actor/global_step_200').glob('*')) if p.is_file()],old_output_directories=sorted(p.name for p in (run/'output').iterdir() if p.is_dir()))
print(json.dumps(x))'''
a=ssh.run('python3 -B -c '+shlex.quote(src),timeout=60);x=dict(returncode=a.returncode,stderr=a.stderr,local_epoch=time.time())
if a.returncode==0:x['remote']=json.loads(a.stdout)
else:x['stdout']=a.stdout
p=Path('outputs/tree-nested-branch-handoff-20261013/OLD3360_CURRENT_PRESERVED_SMALL_METADATA_20261016.json');assert not p.exists();p.write_text(json.dumps(x,indent=2)+'\n')
y=x.get('remote',{});print(json.dumps(dict(returncode=a.returncode,run_exists=y.get('run_exists'),proofs=[{k:v for k,v in p.items() if k!='value'} for p in y.get('proofs',[])],nativefiles=len(y.get('native200_files',[])),HFfiles=len(y.get('HF200_files',[])))))
