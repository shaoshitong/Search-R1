import json,pathlib,subprocess,time,os
A=pathlib.Path('/data/user/jxu957/storage-cleanup-audit-20261026');R='/data/user/jxu957/experiments/';new=R+'dynrpo7b-formal-20261025-cp50-mathfix-r1';old=R+'dynrpo7b-formal-20261023-recoveryfix-r6'
code='''import json,pathlib,hashlib,torch,transformers,faiss
from reproduction.dynrpo7b.restore import source
N=NEW;O=OLD;out={'checkpoints':[],'imports':{'torch':torch.__version__,'transformers':transformers.__version__,'faiss':faiss.__file__}}
for p in [N+'/preparation/dynrpo7b-checkpoints/latest.json',O+'/preparation/dynrpo7b-checkpoints/global_step_50/boundary.json']:
 _,r,cp,origin,parent=source(p)
 probe=r['phase'].get('question_probe_checkpoint')
 out['checkpoints'].append({'receipt':p,'outer':r['completed_outer'],'actor_files_metadata_verified':len(r['actor_ref']['file_bytes']),'rank_Adam':[x['actual_optimizer_writes'] for x in r['actor_ref']['worker_files']],'grounding_manifest_exists':pathlib.Path(r['reward_probe_manifest']['path']).is_file()})
latest=json.loads(pathlib.Path(N+'/preparation/dynrpo7b-checkpoints/latest.json').read_text())
# Locate probe references recursively, without deserializing checkpoint tensors.
def walk(v):
 if isinstance(v,dict):
  if 'path' in v and 'sha256' in v and isinstance(v['path'],str) and v['path'].endswith('.pt') and 'question-probe' in v['path']:
   p=pathlib.Path(v['path']);assert p.is_file();assert hashlib.sha256(p.read_bytes()).hexdigest()==v['sha256']
  for x in v.values():walk(x)
 elif isinstance(v,list):
  for x in v:walk(x)
walk(latest);out['probe_reference_hashes_pass']=True
print(json.dumps(out))
'''.replace('NEW',repr(new)).replace('OLD',repr(old))
p=subprocess.run([new+'/env/bin/python','-c',code],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=90,cwd=new+'/repo')
out={'checked_unix':time.time(),'returncode':p.returncode,'result':json.loads(p.stdout) if p.returncode==0 else p.stdout[-2000:],'stderr':p.stderr[-2000:]}
plan=json.loads((A/'batch9-plan.json').read_text());missing=[];changed=[]
for x in json.loads((A/'batch9-protected-before.json').read_text()):
 try:
  s=os.stat(x['path'])
  if s.st_size!=x['size'] or [s.st_dev,s.st_ino]!=x['inode']:changed.append(x['path'])
 except OSError:missing.append(x['path'])
out.update(protected_missing=missing,protected_changed=changed)
ready=json.loads(pathlib.Path(R+'dynrpo7b-formal-20261023/retrieval/logs/service-ready.json').read_text());out['retrieval_assets']={}
for k in ['corpus_file','index_file']:
 row=ready[k];s=os.stat(row['path']);out['retrieval_assets'][k]={'path':row['path'],'size_match':s.st_size==row['bytes'],'inode_match':s.st_ino==row['inode']}
out['original_models']={}
for root in [R+'topd-20261008/bundle/student_initial',R+'dynrpo-qwen25-7b-probe-warmup-20261019/assets/qwen25-7b-instruct-a09a35458c70']:
 weights=list(pathlib.Path(root).glob('*.safetensors'));out['original_models'][root]={'files':len(weights),'bytes':sum(x.stat().st_size for x in weights)}
(A/'four-roots-verification.json').write_text(json.dumps(out));print(json.dumps(out))
