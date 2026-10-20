"""Authorized CPU select start/status guard; only after actual complete score."""
import pathlib,json,hashlib,fcntl,os,subprocess,time,sys
B=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019')
H=B/'algorithm-improvement-20261021/hard-enrichment-20261021'
S=B/'training-candidate-e4-scoring-20261021/scoring-output/complete.json'
ENTRY_SHA='7b13a4cb6ac126a9c3c972ab66c8458be2af3c6bf9191df132707c11f4415ca1'
PREP_SHA='b3072a3d42180205427d7a22d08c9e8d1520feafc0c69bbbb2bbc845890d503b'
CANDIDATE_SHA='b0118d67ed31187ed0e380cd81d62649852a2806fc9f5d8f16af381f8c4b2832'
MARK='TRAINING_HARD_CPU_SELECTION_20261021'
def read(p):return json.loads(pathlib.Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for raw in iter(lambda:f.read(8*1024**2),b''):h.update(raw)
 return h.hexdigest()
def fp(p):
 p=pathlib.Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def atomic(p,value):
 p=pathlib.Path(p);q=p.with_suffix(p.suffix+'.tmp')
 with q.open('w') as f:json.dump(value,f,sort_keys=True);f.flush();os.fsync(f.fileno())
 q.replace(p)
def alive(pid):
 p=pathlib.Path('/proc')/str(pid)/'cmdline'
 if not p.exists():return False
 value=p.read_bytes().replace(b'\0',b' ').decode(errors='replace')
 return MARK in value and str(H) in value
if not S.exists():
 print(json.dumps(dict(status='WAITING_FOR_ACTUAL_COMPLETE_32768_E4_SCORE',observed_epoch=time.time(),source=str(S),select_launched=False)));sys.exit(0)
with (H/'select-launch.lock').open('a') as guard:
 fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 pub=read(H/'source-publication.json');entry=pathlib.Path(pub['files']['hard_enrichment.py']['path'])
 assert sha(entry)==ENTRY_SHA==pub['files']['hard_enrichment.py']['sha256']
 assert sha(H/'prepare-config.json')==PREP_SHA
 candidate=H/'candidates.json';assert sha(candidate)==CANDIDATE_SHA
 score=read(S);assert score['status']=='E4_PREFIX20_TRAIN_ONLY_CANDIDATE_SCORING_COMPLETE' and score['questions']==32768 and score['trajectories']==163840
 assert score['frozen_parameters_unchanged'] is True and score['optimizer_constructed'] is False
 assert score['actor_optimizer_updates']==score['probe_optimizer_updates']==0 and score['validation_or_final_scored'] is False
 assert score['binding']['candidate_plan']==fp(candidate)
 c=read(H/'prepare-config.json');c.update(candidate_plan=fp(candidate),scoring_complete=fp(S))
 p=H/'select-config.json'
 if p.exists():assert read(p)==c,'Existing immutable selection configuration differs'
 else:atomic(p,c)
 config_ref=fp(p)
 if (H/'selection-complete.json').exists():
  done=read(H/'selection-complete.json');assert done['source_config']==config_ref
  print(json.dumps(dict(status='EXISTING_ACTUAL_CPU_SELECTION_PRESERVED',complete=fp(H/'selection-complete.json'),source_config=config_ref,controls=done['controls'])));sys.exit(0)
 if (H/'select-launch-intent.json').exists():
  intent=read(H/'select-launch-intent.json');state=read(H/'select-controller.json') if (H/'select-controller.json').exists() else None
  assert intent['config']==config_ref and intent['entry']['sha256']==ENTRY_SHA
  print(json.dumps(dict(status='EXISTING_CPU_SELECTION_INTENT_PRESERVED_NO_DUPLICATE',intent=intent,controller=state,process_alive=alive(state['pid']) if state else None,exit=read(H/'select-exit.json') if (H/'select-exit.json').exists() else None)));sys.exit(0)
 intent=dict(status='TRAIN_ONLY_CPU_SELECT_LAUNCH_INTENT',config=config_ref,entry=fp(entry),scoring_complete=fp(S),epoch_time=time.time())
 atomic(H/'select-launch-intent.json',intent)
 child=r'''
# TRAINING_HARD_CPU_SELECTION_20261021
import pathlib,json,hashlib,fcntl,os,subprocess,time,traceback
H=pathlib.Path(ROOT)
def read(p):return json.loads(pathlib.Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for raw in iter(lambda:f.read(8*1024**2),b''):h.update(raw)
 return h.hexdigest()
def atomic(p,v):
 p=pathlib.Path(p);q=p.with_suffix(p.suffix+'.tmp')
 with q.open('w') as f:json.dump(v,f,sort_keys=True);f.flush();os.fsync(f.fileno())
 q.replace(p)
with (H/'select.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 intent=read(H/'select-launch-intent.json');c=read(intent['config']['path']);pid=os.getpid()
 assert sha(intent['config']['path'])==intent['config']['sha256'] and sha(intent['entry']['path'])==intent['entry']['sha256']
 state=dict(status='HASHING_FROZEN_E4_CHECKPOINT_BEFORE_CPU_SELECT',pid=pid,config=intent['config'],entry=intent['entry'],epoch_time=time.time())
 atomic(H/'select-controller.json',state)
 try:
  cp=c['checkpoint'];assert sha(cp['path'])==cp['sha256'] and pathlib.Path(cp['path']).stat().st_size==cp['bytes']
  state.update(status='RUNNING_TRAIN_ONLY_CPU_SELECT',source_E4_full_file_SHA_freshly_verified=True,epoch_time=time.time());atomic(H/'select-controller.json',state)
  env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
  result=subprocess.run(['/usr/bin/python3','-u',intent['entry']['path'],'--config',intent['config']['path'],'--output',str(H),'--action','select'],env=env)
  assert result.returncode==0,'Actual CPU select exited%d'%result.returncode
  complete=read(H/'selection-complete.json');assert complete['status']=='TRAIN_ONLY_HARD_AND_RANDOM22896Q_CLASS_MATCHED_PLANS_COMPLETE'
  assert complete['source_config']==intent['config'] and sha(intent['entry']['path'])==intent['entry']['sha256']
  state.update(status='CPU_SELECT_COMPLETE_PENDING_INDEPENDENT_ACCEPTANCE',exit_code=0,selection_complete_sha256=sha(H/'selection-complete.json'),epoch_time=time.time())
 except Exception as error:
  state.update(status='CPU_SELECT_FAILED',error=repr(error),traceback=traceback.format_exc(),epoch_time=time.time())
  atomic(H/'select-exit.json',state);atomic(H/'select-controller.json',state);raise
 atomic(H/'select-exit.json',state);atomic(H/'select-controller.json',state)
'''.replace('ROOT',repr(str(H)),1)
 with (H/'cpu-select.log').open('ab',buffering=0) as log:
  process=subprocess.Popen(['/usr/bin/python3','-u','-c',child],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 launch=dict(status='ACTUAL_TRAIN_ONLY_CPU_SELECT_STARTED',pid=process.pid,config=config_ref,entry=fp(entry),epoch_time=time.time());atomic(H/'select-launch.json',launch)
 print(json.dumps(launch))
