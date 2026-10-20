import pathlib,json,hashlib,subprocess,fcntl,time
B=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019')
root=B/'algorithm-improvement-20261021/validation-g10-20261021/materialized'
read=lambda p:json.loads(pathlib.Path(p).read_text())
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def checked(item):
 p=pathlib.Path(item['path']);assert sha(p)==item['sha256']
 if 'bytes' in item:assert p.stat().st_size==item['bytes']
 return p
def command(args):
 p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=25)
 assert p.returncode==0,p.stdout+p.stderr
 return p.stdout.strip()
def atomic(p,v):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)
with (root/'sampling-submit.lock').open('a') as guard:
 fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 material=read(root/'materialization.json')
 assert material['status']=='RECIPE_SELECTION_VALIDATION_TWO_FRESH_G5_PLANS_PREPARED_NOT_GENERATED'
 assert material['questions']==2048 and material['planned_real_trajectories']==20480
 assert material['tuning_allowed'] and not material['untouched_final'] and not material['fitting_on_validation_allowed']
 fixed=read(checked(material['fixed_G5_plan']))['fixed_G5']
 assert fixed['summary']['role']=='validation' and fixed['summary']['labels_consulted'] is False
 pub=read(B/'algorithm-data-preparation-code-20261021/source-publication.json')
 assert pub['commit']=='c8b02d3b22b6a3267481e7c632b2f7113b018e93'
 records=[]
 for index,item in enumerate(material['event_configs']):
  label='A' if index==0 else 'B';config_path=checked(item);c=read(config_path);base=config_path.parent
  assert base==root/('event-'+label) and c['base']==str(base)
  assert c['source_commit']==pub['commit'] and c['code_root']==pub['code_root']
  assert c['evaluation_only'] and not c['fitting_authorized'] and not c['actor_updates_authorized'] and c['actor_optimizer_updates']==0
  assert c['same_generation_protocol_as_training'] and c['fixed_view_precommitted']
  assert c['tuning_allowed'] and not c['untouched_final'] and c['capture_prefix_tokens']==20 and c['feature_layer']==26
  for rel,value in c['collector_code_files_sha256'].items():assert sha(pathlib.Path(c['code_root'])/rel)==value
  script=base/'sampling.sbatch';command(['bash','-n',str(script)])
  exports='\n'.join(x for x in script.read_text().splitlines() if x.startswith('export '))
  assert command(['bash','-euc',exports+'\nprintf %s "$QWEN7B_CODE_ROOT"'])==c['code_root']
  name='q7b-validG10-'+label+'-20261021'
  if (base/'job.id').exists():
   job=(base/'job.id').read_text().strip()
   records.append(dict(event=label,status='EXISTING_NOT_RESUBMITTED',job_id=job,accounting=command(['sacct','-X','-n','-P','-j',job,'-o','JobID,State,ExitCode'])))
   continue
  assert not (base/'submission-intent.json').exists(),'Unknown prior outcome; inspect rather than duplicate'
  assert not command(['squeue','-h','-u','mou176','-n',name,'-o','%i|%T|%R']),'Same-name live job requires inspection'
  atomic(base/'submission-intent.json',dict(job_name=name,materialization_sha256=sha(root/'materialization.json'),config=item,fixed_G5=material['fixed_G5_plan'],array='0-3%1',time=time.time()))
  job=command(['sbatch','--hold','--parsable','--job-name='+name,'--array=0-3%1','--time=02:00:00',str(script)]).split(';')[0];assert job.isdigit()
  with (base/'job.id').open('x') as f:f.write(job+'\n')
  record=dict(status='VALIDATION_REAL_G10_ARRAY_HELD',event=label,job_id=job,base=str(base),questions=2048,real_new_trajectories=10240,gpus_per_task=2,max_concurrent_tasks=1,requested_partition='acd_u',requested_qos='normal',source_commit=c['source_commit'],sampling_script=dict(path=str(script),sha256=sha(script)),config=item,fixed_G5=material['fixed_G5_plan'],no_dependency=True,time=time.time())
  atomic(base/'submission.json',record)
  state=command(['squeue','-h','-j',job,'-o','%T|%R']);assert 'PENDING|' in state and 'JobHeldUser' in state
  command(['scontrol','release',job]);record.update(status='VALIDATION_REAL_G10_ARRAY_RELEASED_NOT_YET_VERIFIED',queue=command(['squeue','-h','-j',job,'-o','%i|%j|%T|%P|%R']))
  atomic(base/'submission.json',record);records.append(record)
 print(json.dumps(dict(status='VALIDATION_NEW_G10_SUBMISSIONS_RECONCILED',records=records,time=time.time())))
