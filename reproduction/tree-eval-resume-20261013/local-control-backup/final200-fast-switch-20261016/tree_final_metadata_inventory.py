from pathlib import Path
import sys,os,shlex,json,time,hashlib
sys.path.insert(0,'work/tree-speed-repair-20261008');import ssh_retained18 as ssh
os.environ['TREE_MGMT_CONTROL_PATH']='/tmp/tree-mgmt-3fw9a1ab/control'
b=Path('outputs/tree-nested-branch-handoff-20261013');at=json.loads((b/'TREE_TRACE_RETENTION_THROUGH200_ACTUAL_ATTEMPT.json').read_text())
program_sha=at['remote_program_sha256']
src=r'''import json,time,hashlib,ast,base64,zlib
from pathlib import Path
from omegaconf import OmegaConf
import pyarrow.parquet as pq
ROOT=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo')
r=ROOT/'runs/tree-nested-branch-fresh200-not-slurm;systemd:778b2b43dad90fc0c1c5d95d77c2326286011b73'
parent=ROOT/'runs/tree-nested-branch-fresh200-not-slurm;systemd:ba9f6627ec9a4e9749830d7b6acca53f6feec6b4'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def stat(p):
 if not p.is_file():return dict(path=str(p),exists=False)
 z=p.stat();return dict(path=str(p),exists=True,bytes=z.st_size,mtime_ns=z.st_mtime_ns)
x=dict(epoch=time.time(),trace_program_sha256=PROGRAM_SHA,trace_process_observations=[],checkpoints=[])
for _ in range(2):
 matches=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   cmd=(p/'cmdline').read_bytes().split(b'\0')
   if b'-c' not in cmd:continue
   code=cmd[cmd.index(b'-c')+1].decode()
   if 'exec(zlib.decompress(base64.b64decode(' not in code:continue
   tree=ast.parse(code)
   for n in ast.walk(tree):
    if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='b64decode' and n.args and isinstance(n.args[0],ast.Constant):
     raw=zlib.decompress(base64.b64decode(n.args[0].value))
     if hashlib.sha256(raw).hexdigest()==PROGRAM_SHA:matches.append(dict(pid=int(p.name),stat=(p/'stat').read_text(),cgroup=(p/'cgroup').read_text(),cmdline_sha256=hashlib.sha256(b'\0'.join(cmd)).hexdigest()))
  except (OSError,ValueError,UnicodeError,SyntaxError,IndexError,zlib.error):pass
 x['trace_process_observations'].append(dict(epoch=time.time(),matches=matches));time.sleep(.25)
p=r/'resolved.yaml';raw=p.read_bytes();x['resolved_config']=dict(path=str(p),bytes=len(raw),sha256=sha(p),raw=raw.decode())
c=OmegaConf.load(p);OmegaConf.resolve(c)
x['actual_parquet_metadata']={}
for key in ['train_files','val_files','final_val_files']:
 value=c.data.get(key);paths=[value] if isinstance(value,str) else list(value or [])
 x['actual_parquet_metadata'][key]=[dict(path=str(q),metadata_rows=pq.ParquetFile(str(q)).metadata.num_rows,bytes=Path(str(q)).stat().st_size) for q in paths]
model=Path(str(c.actor_rollout_ref.model.path));x['official_model_small_metadata']=[]
for name in ['config.json','tokenizer_config.json','MODEL_IDENTITY.json','model-identity.json','download-manifest.json','revision.txt']:
 p=model/name
 if p.is_file() and p.stat().st_size<65536:x['official_model_small_metadata'].append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),raw=p.read_text()))
for outer in [5,50,100,150,200]:
 run=parent if outer==5 else r;n=run/'checkpoints/training_state'/('global_step_'+str(outer));hf=run/'checkpoints/actor'/('global_step_'+str(outer));p=n/'manifest.json';entry=dict(outer=outer,native_manifest=stat(p),native_files=[stat(q) for q in sorted(n.glob('*')) if q.is_file()],HF_files=[stat(q) for q in sorted(hf.glob('*')) if q.is_file()])
 if p.is_file():entry['native_manifest'].update(sha256=sha(p),value=json.loads(p.read_text()))
 x['checkpoints'].append(entry)
print(json.dumps(x,ensure_ascii=False))'''.replace('PROGRAM_SHA',repr(program_sha))
t=time.time();a=ssh.run('/dataset/shaoshitong/search-r1-rl/tree-grpo/env/bin/python -B -c '+shlex.quote(src),timeout=90)
x=dict(returncode=a.returncode,stderr=a.stderr,wall_seconds=time.time()-t,local_epoch=time.time())
if a.returncode==0:x['remote']=json.loads(a.stdout)
else:x['stdout']=a.stdout
p=b/'FINAL200_CURRENT_METADATA_AND_TRACE_PROCESS_READBACK.json';assert not p.exists();p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
if a.returncode==0:
 q=b/'FINAL200_CURRENT_RESOLVED.yaml';assert not q.exists();q.write_text(x['remote']['resolved_config']['raw']);print(json.dumps(dict(returncode=0,path=str(p),trace=x['remote']['trace_process_observations'],parquet=x['remote']['actual_parquet_metadata'],checkpoints=[dict(outer=q['outer'],native_exists=q['native_manifest']['exists'],native_files=len(q['native_files']),hf_files=len(q['HF_files'])) for q in x['remote']['checkpoints']]),ensure_ascii=False))
else:print(json.dumps(x))
