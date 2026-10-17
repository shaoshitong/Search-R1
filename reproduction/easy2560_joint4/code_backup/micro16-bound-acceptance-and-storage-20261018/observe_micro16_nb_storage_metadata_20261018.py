import json,os,stat,subprocess,time,socket
from pathlib import Path
b=Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-micro16-after730080-20261018');assert b.resolve()==b and b.stat().st_uid==os.getuid()==14090
started=time.time();argv=['du','-x','-B1','--max-depth=3','--',str(b)];p=subprocess.run(argv,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=45);finished=time.time();du=[]
for line in p.stdout.splitlines():
 size,path=line.split('\t',1);du.append({'allocated_bytes':int(size),'path':path})
def small(path):
 if not path.is_file() or path.is_symlink():return None
 assert path.stat().st_size<2*1024**2
 return json.loads(path.read_text())
run=small(b/'logs/run-manifest.json');assert run['run_id']=='dfe9c94084bb48b2a9bb7b60f9d882da' and run['source_commit']=='8a5bac94b8230a99b81977f5290589e3d3708dfe'
paths={}
for name in ('logs/retrieval-staging-claim.json','logs/retrieval-assets.json','logs/corpus-cache-claim.json','logs/retrieval-admission.json','preparation/environment-reuse.json'):
 j=small(b/name)
 if j is not None:
  if name.endswith('retrieval-admission.json'):j={k:j.get(k) for k in ('accepted','run_id','shard_count','ntotal','dimension','use_float16','cache','cpu_index_release')}
  paths[name]=j
links=[]
for x in b.iterdir():
 if x.is_symlink():links.append({'relative_path':x.name,'link_target':os.readlink(x),'link_allocated_bytes':x.lstat().st_blocks*512})
files=[];walk_errors=[]
for top in ('artifacts','logs'):
 for directory,dirs,names in os.walk(b/top,followlinks=False,onerror=lambda e:walk_errors.append(str(e))):
  dirs[:]=[n for n in dirs if not (Path(directory)/n).is_symlink()]
  for name in names:
   path=Path(directory)/name;s=path.lstat()
   if not stat.S_ISREG(s.st_mode):continue
   rel=str(path.relative_to(b));files.append({'path':rel,'bytes':s.st_size,'allocated_bytes':s.st_blocks*512,'links':s.st_nlink,'device':s.st_dev,'inode':s.st_ino,'mtime_ns':s.st_mtime_ns})
raw_names={'0.pt','512.pt','1024.pt','1536.pt','2048.pt'};categories={};pools={}
for row in files:
 rel=Path(row['path']);category='other_artifacts_or_logs'
 if str(rel).startswith('artifacts/mixed-candidate-pools/'):
  category='candidate_five_raw_tensors' if rel.name in raw_names else 'retained_candidate_rng_and_pool_evidence'
  pool=rel.parts[2];x=pools.setdefault(pool,{'raw':[],'retained':[]});x['raw' if rel.name in raw_names else 'retained'].append(row)
 elif str(rel).startswith('artifacts/behavior-traces/'):category='selected_private_behavior_trace'
 elif str(rel).startswith('artifacts/mixed-selection/'):category='selection_audit_json'
 elif str(rel).startswith('logs/behavior-probability'):category='native_lp_gate_hash_journal'
 elif str(rel).startswith('logs/'):category='other_logs'
 x=categories.setdefault(category,{'file_count':0,'allocated_bytes':0,'logical_bytes':0,'multilink_allocated_bytes':0});x['file_count']+=1;x['allocated_bytes']+=row['allocated_bytes'];x['logical_bytes']+=row['bytes'];x['multilink_allocated_bytes']+=row['allocated_bytes'] if row['links']>1 else 0
for pool,x in pools.items():
 x['raw_allocated_bytes']=sum(r['allocated_bytes'] for r in x['raw']);x['retained_allocated_bytes']=sum(r['allocated_bytes'] for r in x['retained']);x['progress']=small(b/'artifacts/mixed-candidate-pools'/pool/'progress.json');x['selection_failure']=small(b/'artifacts/mixed-candidate-pools'/pool/'selection-failure.json')
updates=[]
q=b/'logs/actual-updates.jsonl'
if q.exists():
 for line in q.read_text().splitlines():
  r=json.loads(line);updates.append({k:r.get(k) for k in ('step','attempt_id','attempt_index','actual_optimizer_writes')})
for x in pools.values():
 if x['progress']:
  x['progress']['complete_ordinals_count']=len(x['progress'].pop('complete_ordinals',[]));reserved=x['progress'].get('reserved');x['progress']['reserved_count']=len(reserved) if isinstance(reserved,(dict,list)) else reserved
  if isinstance(reserved,(dict,list)):x['progress'].pop('reserved',None)
q=b/'preparation/candidate-raw-maintenance-runner/progress.json';runner=small(q)
fields=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
print(json.dumps({'role':'one_bounded_current_NB_allocated_metadata_du','base':str(b),'started_unix':started,'du_finished_unix':finished,'finished_unix':time.time(),'host':socket.gethostname(),'uid':os.getuid(),'observer_birth':{'pid':os.getpid(),'start_ticks':int(fields[19]),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'pid_namespace':os.readlink('/proc/self/ns/pid')},'du_command':argv,'du_returncode':p.returncode,'du_stderr':p.stderr,'du_rows':du,'symlink_targets_not_followed':links,'run_fields':{k:run.get(k) for k in ('run_id','source_commit','checkpoint_dir','train_log','initial_model','local_retrieval_assets','ray_tmp_dir','retrieval_asset_paths','retriever_gpu_lib','retrieval_encoder')},'path_manifests':paths,'file_categories':categories,'pools':pools,'completed_updates':updates,'largest_30_files_in_artifacts_logs':sorted(files,key=lambda r:r['allocated_bytes'],reverse=True)[:30],'all_artifacts_logs_file_stats':files,'metadata_walk_errors':walk_errors,'raw_runner_progress':runner,'tensor_contents_read':False,'new_payload_sha_executed':False,'other_experiments_or_shared_roots_scanned':False,'full_personal_quota_measured':False},sort_keys=True))
