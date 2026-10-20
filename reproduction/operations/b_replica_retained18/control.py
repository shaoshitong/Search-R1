"""Local orchestration through the already successful pinned controller only."""
import argparse,base64,gzip,hashlib,io,json,pathlib,subprocess,sys,tarfile,time
REPO=pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0,'/Users/shaoshitong/Documents/Search-G1/prefix1024-diversity512-20261016')
if __package__:
 from .observation_transport import remote
else:
 from observation_transport import remote
OUT=pathlib.Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2/outputs/b-replica-retained18-20261018')
PHYSICAL='/mnt/data/shaoshitong/search-g1-prefix-selection200-reference-replica-b-own-20261018-cache1-kvcap1-blocks1'
CANONICAL='/mnt/data/sihe-share5/workspace/training_data_node_10/shaoshitong/'+PHYSICAL.rsplit('/',1)[1]
SOURCE='/mnt/data/shaoshitong/search-g1-prefix-selection200-reference-replica-b-20261018-d9e5385ebec2-cache-repair1-kv-profile-repair1-actor-memory-repair1-h10011-r1/preparation/launch-config.json'
COMMON='''import pathlib,json,subprocess,time,os,sys,hashlib
p=pathlib.Path(PHYSICAL);canonical=CANONICAL
wrapper=p/'repo/reproduction/operations/b_replica_retained18/private_paths.sh'
def inside(code):
 command=['unshare','--mount','--propagation','private','/bin/bash',str(wrapper),'/dataset/shaoshitong/Search-G1-common8/env/bin/python','-c',code]
 r=subprocess.run(command,capture_output=True,text=True,timeout=180)
 return dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
'''
def main():
 a=argparse.ArgumentParser();a.add_argument('phase',choices=['prepare','repair_paths','repair_cache','repair_kv','submit','observe']);args=a.parse_args()
 OUT.mkdir(exist_ok=True)
 s=COMMON.replace('PHYSICAL',repr(PHYSICAL)).replace('CANONICAL',repr(CANONICAL))
 if args.phase in ('prepare','repair_cache','repair_kv'):
  commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
  tracked=subprocess.check_output(['git','ls-files','-z'],cwd=REPO).decode().strip('\0').split('\0')
  manifest=[dict(path=f,sha256=hashlib.sha256((REPO/f).read_bytes()).hexdigest())for f in tracked]
  buffer=io.BytesIO()
  with tarfile.open(fileobj=buffer,mode='w')as tar:
   for f in tracked:
    if f.startswith('reproduction/operations/b_replica_retained18/')or f in ('reproduction/prefix20_online/retriever.py','reproduction/prefix_selection200/rollout.py','reproduction/prefix_selection200/eval.py'):tar.add(REPO/f,arcname=f,recursive=False)
  archive=gzip.compress(buffer.getvalue(),compresslevel=6)
  packed_manifest=base64.b64encode(gzip.compress(json.dumps(manifest).encode())).decode()
  source_config=SOURCE if args.phase=='prepare' else '/mnt/data/shaoshitong/search-g1-prefix-selection200-reference-replica-b-own-20261018/preparation/launch-config.json'
  if args.phase=='repair_kv':source_config='/mnt/data/shaoshitong/search-g1-prefix-selection200-reference-replica-b-own-20261018-cache1-kvcap1/preparation/launch-config.json'
  s+='''\nimport io,tarfile,base64,gzip
assert not p.exists(),'Existing deployment must be reconciled, never overwritten'
p.mkdir();(p/'repo').mkdir();(p/'preparation').mkdir();(p/'logs').mkdir()
raw=gzip.decompress(base64.b64decode(ARCHIVE))
with tarfile.open(fileobj=io.BytesIO(raw))as tar:
 for f in tar.getmembers():
  assert not f.name.startswith('/') and '..'not in pathlib.Path(f.name).parts
  assert f.isdir()or f.isfile(),f.name
 tar.extractall(p/'repo')
source=pathlib.Path(SOURCE).parents[1]/'repo'
for row in json.loads(gzip.decompress(base64.b64decode(MANIFEST))):
 relative=pathlib.Path(row['path']);assert not relative.is_absolute()and '..'not in relative.parts
 target=p/'repo'/relative
 if not target.exists():
  original=source/relative;assert original.is_file()and not original.is_symlink(),str(original)
  data=original.read_bytes();assert hashlib.sha256(data).hexdigest()==row['sha256'],str(relative)
  target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
 assert hashlib.sha256(target.read_bytes()).hexdigest()==row['sha256'],str(relative)
sys.path.insert(0,str(p/'repo'))
from reproduction.operations.b_replica_retained18.deploy import prepare
c=prepare(str(p),canonical,SOURCE,COMMIT)
files=[dict(path=str(f.relative_to(p/'repo')),sha256=hashlib.sha256(f.read_bytes()).hexdigest())for f in(p/'repo').rglob('*')if f.is_file()]
(p/'preparation/source-full-sha.json').write_text(json.dumps(dict(commit=COMMIT,files=files)))
check=inside("import sys,json;sys.path.insert(0,"+repr(canonical+'/repo')+");from reproduction.prefix_selection200.launch import config; c=config("+repr(canonical)+");print(json.dumps(dict(run_id=c['run_id'],mode=c['selection_probe_experiment'],base=c['base'],source_commit=c['source_commit'])))")
print(json.dumps(dict(config=c,source_files=len(files),validation=check)))
'''.replace('ARCHIVE',repr(base64.b64encode(archive).decode())).replace('MANIFEST',repr(packed_manifest)).replace('SOURCE',repr(source_config)).replace('COMMIT',repr(commit))
  if args.phase in ('repair_cache','repair_kv'):
   s=s.replace('from reproduction.operations.b_replica_retained18.deploy import prepare\nc=prepare(str(p),canonical,'+repr(source_config)+','+repr(commit)+')', 'from reproduction.operations.b_replica_retained18.cache_retry import prepare\nprepare('+repr(str(pathlib.Path(source_config).parents[1]))+',str(p),canonical,'+repr(commit)+')\nc=json.loads((p/"preparation/launch-config.json").read_text())')
  if args.phase=='repair_kv':s=s.replace('from reproduction.operations.b_replica_retained18.cache_retry import prepare','from reproduction.operations.b_replica_retained18.kv_retry import prepare')
 elif args.phase=='repair_paths':
  commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
  files={str(f.relative_to(REPO)):base64.b64encode(f.read_bytes()).decode()for f in(REPO/'reproduction/operations/b_replica_retained18').glob('*')if f.is_file()}
  s+='''
import base64
assert not(p/'preparation/submit-intent.json').exists(),'No edits after submission'
assert not(p/'actor').exists()and not(p/'probe').exists()
backup=p/'preparation/before-private-path-repair1';backup.mkdir(exist_ok=False)
cpath=p/'preparation/launch-config.json';(backup/'launch-config.json').write_bytes(cpath.read_bytes())
for name,data in FILES.items():
 f=p/'repo'/name;(backup/f.name).write_bytes(f.read_bytes());f.write_bytes(base64.b64decode(data))
c=json.loads(cpath.read_text());c['source_commit']=COMMIT;cpath.write_text(json.dumps(c,sort_keys=True))
rows=[dict(path=str(f.relative_to(p/'repo')),sha256=hashlib.sha256(f.read_bytes()).hexdigest())for f in(p/'repo').rglob('*')if f.is_file()and'__pycache__'not in f.parts]
(p/'preparation/source-full-sha-after-private-path-repair1.json').write_text(json.dumps(dict(commit=COMMIT,files=rows)))
print(json.dumps(dict(commit=COMMIT,files=len(rows),check=inside("import sys,json;sys.path.insert(0,"+repr(canonical+'/repo')+");from reproduction.prefix_selection200.launch import config;print(json.dumps(config("+repr(canonical)+")['run_id']))"))))
'''.replace('FILES',repr(files)).replace('COMMIT',repr(commit))
 elif args.phase=='submit':
  s+='''\ncode="import sys,json;sys.path.insert(0,"+repr(canonical+'/repo')+");from reproduction.prefix_selection200.launch import submit;print(json.dumps(submit("+repr(canonical)+")))"
print(json.dumps(inside(code)))
'''
 else:
  s+='''\ndef read(f):return json.loads(f.read_text())if f.exists()else None
c=read(p/'preparation/launch-config.json');j=read(p/'latest-joint.json');stage=read(p/'logs/pipeline-status.json');receipt=read(p/'preparation/submit-receipt.json');leader=None
if receipt:
 who=receipt.get('leader',{});pid=who.get('pid');proc=pathlib.Path('/proc')/str(pid)
 if proc.exists():leader=dict(pid=pid,stat=(proc/'stat').read_text(),cmdline=(proc/'cmdline').read_bytes().replace(b'\\0',b' ').decode())
logs=[]
for f in sorted((p/'logs').glob('*.log'),key=lambda f:f.stat().st_mtime,reverse=True)[:5]:
 with f.open('rb')as stream:stream.seek(max(0,f.stat().st_size-2500));tail=stream.read().decode(errors='replace')
 logs.append(dict(path=str(f),bytes=f.stat().st_size,mtime=f.stat().st_mtime,tail=tail))
outer=(stage or{}).get('outer',0);facts={}
if j:
 facts=dict(outer=j['outer'],actor=j['actor']['actual_adam_steps_per_rank'],probe=j['probe']['actual_adam_steps_per_rank'],phase_timings=j.get('phase_timings'),selection_metrics=j.get('selection_metrics'))
progress={}
for f in (p/'outers'/('outer-%06d'%(outer+1 if str((stage or{}).get('stage','')).startswith('CANDIDATE')else max(1,outer)))).rglob('*progress*.json'):
 if f.stat().st_size<10000:progress[str(f.relative_to(p))]=read(f)
print(json.dumps(dict(observed_unix=time.time(),run_id=c['run_id'],base=str(p),canonical=canonical,joint=facts,stage=stage,leader=leader,receipt=receipt,logs=logs,progress=progress,cleanup=read(p/'logs/cleanup-status.json'),complete=read(p/'complete.json'))))
'''
 compile(s,'B-remote-'+args.phase,'exec')
 e=OUT/(args.phase+'-'+str(time.time_ns())+'-transport.json')
 raw=remote(s,e,target='h100-10',established_controller=True);v=json.loads(raw)
 path=e.with_name(e.name.replace('-transport','-actual'));path.write_text(json.dumps(v,indent=2));(OUT/'LATEST.json').write_text(json.dumps(dict(phase=args.phase,evidence=str(path),actual=v),indent=2))
 if args.phase in ('prepare','repair_cache','repair_kv'):
  (OUT/'CURRENT_CONFIG.json').write_text(json.dumps(v['config'],indent=2));print(json.dumps(dict(evidence=str(path),validation=v['validation'],files=v['source_files'])))
 else:print(json.dumps(dict(evidence=str(path),actual=v)))
if __name__=='__main__':main()
