import json,subprocess,pathlib,time
cfg='/Users/shaoshitong/Documents/remote_codex_control/access/hpc3-20261008/ssh_config';sock='/tmp/grounding-eval-20261019-501/hpc3-cm.sock'
code=r'''
import pathlib,json,subprocess,time,re
b=pathlib.Path('/data/user/jxu957/experiments/grounding-eval-strict-20261019');out={'at':time.time(),'jobs':{},'gpu_nodes':{}}
def cmd(a,timeout=15):
 try:
  r=subprocess.run(a,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=timeout);return {'rc':r.returncode,'stdout':r.stdout[-15000:],'stderr':r.stderr[-1000:]}
 except Exception as e:return {'error_type':type(e).__name__}
def small(p,limit=100000):
 if not p.exists():return None
 if p.stat().st_size>limit:return {'exists':True,'bytes':p.stat().st_size}
 try:return json.loads(p.read_text())
 except:return {'exists':True,'bytes':p.stat().st_size,'parse_failed':True}
nodes=set()
for jid,name in [('737435','topd-alpha01-v4'),('737436','opd-alpha1-v4')]:
 root=b/name;pj=small(root/'pipeline-job.json');nj=small(root/'normal-job.json');r={'root':str(root),'squeue':cmd(['squeue','-h','-j',jid,'-o','%i|%T|%N|%b|%C|%m|%M|%R']),'scontrol':cmd(['scontrol','show','job',jid,'-d','-o']),'sacct':cmd(['sacct','-j',jid,'-nP','--format=JobID,State,ExitCode,Elapsed,AllocTRES,NodeList'])}
 for x in re.findall(r'NodeList=([A-Za-z0-9_.-]+)',r['scontrol'].get('stdout','')):
  if x not in ['None','null']:nodes.add(x)
 nr=pathlib.Path(nj['output_dir']);pr=pathlib.Path(pj['pipeline_output']);cr=pathlib.Path(pj['cf_output']);r['paths']={'normal_root':str(nr),'pipeline_root':str(pr),'cf_root':str(cr)}
 r['pipeline']={n:small(pr/n) for n in ['worker.json','progress.json','normal-exit.json','counterfactual-exit.json','failed.json','complete.json']}
 r['normal']={n:small(nr/n) for n in ['worker.json','normal-progress.json','normal-error.json','normal-complete.json','retriever-process.json','retriever-exit.json']}
 r['normal_executions']={p.name:small(p) for p in nr.glob('normal-execution-*.json')}
 batches=list((nr/'normal-batches').glob('batch-*.json'));stats={'batch_files':len(batches),'bytes':sum(p.stat().st_size for p in batches),'records':0,'token_totals':{},'latest_mtime':max([p.stat().st_mtime for p in batches],default=None)}
 for p in batches:
  try:
   d=json.loads(p.read_text());stats['records']+=len(d['records'])
   for row in d['records']:
    for k in ['model_generated_tokens','raw_generated_tokens','observation_tokens','initial_prompt_tokens','trajectory_total_tokens','cumulative_model_input_tokens']:
     v=row.get(k)
     if isinstance(v,(int,float)):stats['token_totals'][k]=stats['token_totals'].get(k,0)+v
  except Exception as e:stats.setdefault('read_errors',[]).append({'file':p.name,'error_type':type(e).__name__})
 r['actual_normal_files']=stats
 r['cf_progress']={n:small(cr/n) for n in ['progress.json','summary.json','verified-summary.json']}
 r['logs']={}
 for p in [pr/('slurm-'+jid+'.log'),pr/'normal.log',nr/'retriever.log',pr/'counterfactual.log']:
  if not p.exists():continue
  with p.open('rb') as f:
   f.seek(max(0,p.stat().st_size-12000));lines=f.read().decode(errors='replace').splitlines()
  r['logs'][str(p)]={'bytes':p.stat().st_size,'mtime':p.stat().st_mtime,'tail':lines[-16:],'errors':[l[:1200] for l in lines if any(s in l for s in ['Traceback','Error','CUDA out of memory','FAILED','Exception','Killed'])][-12:]}
 out['jobs'][jid]=r
for n in []: # Compute directSSH failed strict hostkey; do not retry or bypass.
 out['gpu_nodes'][n]=cmd(['ssh','-T','-oBatchMode=yes','-oStrictHostKeyChecking=yes','-oConnectTimeout=6',n,'nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits'],12)
print(json.dumps(out))
'''
r=subprocess.run(['ssh','-T','-F',cfg,'-S',sock,'-oUser=mou176','-oBatchMode=yes','hpc3','/usr/bin/python3 -'],input=code,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=80);d={'transport_rc':r.returncode,'stderr':r.stderr[-1000:]}
try:d['audit']=json.loads(r.stdout.strip().splitlines()[-1])
except:d['parse_failed']=True;d['stdout_tail']=r.stdout[-1000:]
p=pathlib.Path('outputs/grounding-eval-20261019')/('hpc3-topd-opd-observation-'+time.strftime('%Y%m%dT%H%M%S')+'.json');p.write_text(json.dumps(d,indent=2));print(p)
for j,v in d.get('audit',{}).get('jobs',{}).items():print(j,json.dumps({'squeue':v['squeue'],'normal_files':v['actual_normal_files'],'normal_progress':v['normal']['normal-progress.json'],'pipeline_progress':v['pipeline']['progress.json'],'errors':{k:v['errors'] for k,v in v['logs'].items() if v['errors']},'tails':{k:v['tail'][-4:] for k,v in v['logs'].items()}}))
print(json.dumps(d.get('audit',{}).get('gpu_nodes',{})))
