"""Code-only isolated data deployment; query the existing intent, never replay."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

from reproduction.prefix20_group_joint100.control import archive_for_commit
from reproduction.prefix_selection200.control import deployment_script
from reproduction.h1008_qprobe.established_controller_transport import remote


def deploy_code(config,commit,reuse_commit,reuse_repo,role='data',stage_only=False):
    assert role in ('data','fit') and config['probe20k_role']==role
    deployment=deployment_script(config['base'],commit,config,archive_for_commit(commit,reuse_commit),reuse_commit,reuse_repo)
    before=''
    old=config['probe20k_study'].get('replace_pending')
    recovery=config['probe20k_study'].get('resume_dataset_base')if role=='data'else None
    if recovery:
        before=r'''
import json,pathlib,subprocess
s=SPEC
proof=json.loads(pathlib.Path(s['prior_native_terminal_proof']).read_text())
released=json.loads(pathlib.Path(s['prior_release_proof']).read_text())
assert proof['cleanup']['owned_births_all_gone']and not released['remaining_original_births']and released['original_birth_count']>0
r=subprocess.run(['scontrol','show','job',s['prior_native_job'],'-o'],text=True,capture_output=True,timeout=30);assert r.returncode==0
native=dict(item.split('=',1)for item in r.stdout.split()if '='in item)
assert native['JobState']in('FAILED','CANCELLED','OUT_OF_MEMORY','COMPLETED')and native['Command']==s['resume_dataset_base']+'/preparation/native-probe20k-data.sh'
'''.replace('SPEC',repr(config['probe20k_study']))
    prior=config['probe20k_study'].get('prior_stopped_fit_base')if role=='fit'else None
    if prior:
        before=r'''
import pathlib,json,subprocess
s=SPEC;source=pathlib.Path(s['prior_stopped_fit_base'])
p=json.loads((source/s.get('prior_stopped_fit_proof_name','preparation/human-stop-A-run-BC5-terminal-once.json')).read_text())
assert p['original_births_all_gone']and p['no_live_own_gpu']and p['base']==str(source)
r=subprocess.run(['scontrol','show','job',s['prior_stopped_fit_job'],'-o'],capture_output=True,text=True,timeout=20)
expected_states=(s['prior_stopped_fit_terminal_state'],)if s.get('prior_stopped_fit_terminal_state')else('CANCELLED','COMPLETED')
assert all(state in ('CANCELLED','COMPLETED','FAILED')for state in expected_states)
if r.returncode==0:
 assert any('JobState='+state in r.stdout for state in expected_states)and str(source/'preparation/native-probe20k-fit.sh')in r.stdout
else:
 # Slurm may purge an already sealed terminal job. Its original native record
 # and two owned-resource-release records remain authoritative, not a new job.
 assert 'Invalid job id' in r.stderr or 'Invalid job id' in r.stdout,(r.stdout,r.stderr)
 archived=p.get('native','');assert 'JobId='+s['prior_stopped_fit_job']+' 'in archived
 assert any('JobState='+state in archived for state in expected_states)and str(source/'preparation/native-probe20k-fit.sh')in archived
 if 'JobState=COMPLETED'in archived:assert 'ExitCode=0:0'in archived
 assert p['job_id']==s['prior_stopped_fit_job']and len(p['release'])==2
 assert all(not v['remaining_original_births']and not v['owned_gpu']and not v['cgroups']for v in p['release'])
'''.replace('SPEC',repr(config['probe20k_study']))
    if old:
        before=r'''
import pathlib,json,sys,subprocess,time
old=OLD;root=pathlib.Path(old['base'])
r=subprocess.run(['scontrol','show','job',str(old['job_id']),'-o'],text=True,capture_output=True,timeout=30);assert r.returncode==0,r.stderr
native=dict(item.split('=',1)for item in r.stdout.split()if '='in item)
assert native['Command']==str(root/'preparation/native-probe20k-data.sh')
if native['JobState']=='RUNNING':
 print(json.dumps(dict(status='ORIGINAL_DATA_JOB_NOW_RUNNING_NO_REPLACEMENT',job_id=old['job_id'])));raise SystemExit
record=root/'preparation/replaced-pending-for-cpu-memory-profile.json'
if not record.exists():
 assert native['JobState']=='PENDING'and native['RunTime']=='00:00:00'
 cancelled=subprocess.run(['scancel',str(old['job_id'])],text=True,capture_output=True,timeout=30);assert cancelled.returncode==0,cancelled.stderr
 with record.open('x')as f:json.dump(dict(decision_unix=time.time(),job_id=str(old['job_id']),native_before=native,
  reason='Latest live20k EMA task; replace only never-started own pending job to fit CPU/memory; physical8GPU/science unchanged',
  later_task_authorization_active=True,training_was_not_started=True,new_cpus=48,new_memory_gib=256),f)
else:assert native['JobState']=='CANCELLED'and json.loads(record.read_text())['job_id']==str(old['job_id'])
'''.replace('OLD',repr(old))
    code=(before+'import pathlib,json,sys\nb=pathlib.Path('+repr(config['base'])+');\n'
        'if not b.exists():\n exec(compile('+repr(deployment)+',"isolated_probe20k_deploy","exec"),{})\n'
        'else:\n assert (b/"preparation/deploy-receipt.json").exists(),"Reconcile existing deployment intent first"\n'
        'c=json.loads((b/"preparation/launch-config.json").read_text());assert c["source_commit"]=='+repr(commit)+'\n')
    if stage_only:
        return code+'print(json.dumps(dict(deploy=json.loads((b/"preparation/deploy-receipt.json").read_text()),submit=None,stage_only=True)))\n'
    return(code+
        'sys.path.insert(0,str(b/"repo"));from reproduction.prefix_selection200.'+('probe20k_dataset'if role=='data'else'probe20k_fit')+' import submit\n'
        'print(json.dumps(dict(deploy=json.loads((b/"preparation/deploy-receipt.json").read_text()),submit=submit(b))))\n')


def observation_code(base):
    return r'''
import pathlib,json,subprocess,sys,time
b=pathlib.Path(BASE);sys.path.insert(0,str(b/'repo'))
from reproduction.common8.native_allocation import fields
def read(p):return json.loads(p.read_text())if p.exists()else None
receipt=read(b/'preparation/submit-receipt.json');job=receipt['job_id']
r=subprocess.run(['scontrol','show','job',job,'-o'],capture_output=True,text=True,timeout=30)
blocks=[];config=read(b/'preparation/launch-config.json');previous=config['probe20k_study'].get('resume_dataset_base')
for outer in(52,53,54):
 root=b/'outers'/('outer-%06d'%outer);value=dict(outer=outer)
 if previous:
  old=pathlib.Path(previous)/'outers'/('outer-%06d'%outer)
  if(old/'prefix/complete.json').exists()and(old/'full/complete.json').exists():root=old;value['reused_complete']=True
 value['root']=str(root)
 for phase in('prefix','full'):
  rows=[read(root/phase/('rank-%d-progress.json'%rank))for rank in range(8)]
  value[phase+'_completed']=sum((row or{}).get('completed_trajectories',0)for row in rows)
  value[phase+'_complete']=(root/phase/'complete.json').exists()
 blocks.append(value)
errors={}
for p in(b/'logs').glob('*.log'):
 with p.open('rb')as f:
  f.seek(max(0,p.stat().st_size-16000));tail=f.read().decode(errors='replace')
 selected=[line for line in tail.splitlines()[-35:]if any(v in line for v in('Traceback','Error','Exception','failed'))]
 if selected:errors[p.name]='\n'.join(selected)
print(json.dumps(dict(observed_unix=time.time(),base=str(b),job_id=job,
 native=fields(r.stdout)if r.returncode==0 else dict(unavailable=r.stderr),
 stage=read(b/'logs/pipeline-status.json'),progress=read(b/'logs/dataset-progress.json'),
 complete=read(b/'dataset-complete.json'),cleanup=read(b/'logs/cleanup-status.json'),
 blocks=blocks,error_tails=errors,training_modified=False)))
'''.replace('BASE',repr(base))


def fit_observation_code(base):
    return r'''
import pathlib,json,subprocess,sys,time
b=pathlib.Path(BASE);sys.path.insert(0,str(b/'repo'))
from reproduction.common8.native_allocation import fields
def read(p):return json.loads(p.read_text())if p.exists()else None
receipt=read(b/'preparation/submit-receipt.json');job=receipt['job_id']
r=subprocess.run(['scontrol','show','job',job,'-o'],capture_output=True,text=True,timeout=30)
arms={}
config=read(b/'preparation/launch-config.json')
for name in [a['name']for a in config['probe20k_study']['arms']]:
 root=b/'arms'/name;last=[]
 for rank in range(8):
  p=root/('rank-%d-updates.jsonl'%rank)
  if not p.exists():last.append(None);continue
  with p.open('rb')as f:
   f.seek(max(0,p.stat().st_size-8192));lines=f.read().decode().splitlines()
  try:last.append(json.loads(lines[-1]))
  except (ValueError,IndexError):last.append(None)
 metrics=[read(p)for p in sorted(root.glob('epoch-*-metrics.json'))]
 arms[name]=dict(latest=read(root/'latest.json'),progress=read(root/'progress.json'),
  metrics=metrics,initialization=read(root/'initialization.json'),complete=read(root/'complete.json'),rank_last_updates=last)
errors={}
for p in(b/'logs').glob('*.log'):
 with p.open('rb')as f:
  f.seek(max(0,p.stat().st_size-20000));tail=f.read().decode(errors='replace')
 selected=[line for line in tail.splitlines()[-45:]if any(v in line for v in('Traceback','Error','Exception','failed'))]
 if selected:errors[p.name]='\n'.join(selected)
print(json.dumps(dict(observed_unix=time.time(),base=str(b),job_id=job,
 native=fields(r.stdout)if r.returncode==0 else dict(unavailable=r.stderr),stage=read(b/'logs/pipeline-status.json'),
 arms=arms,complete=read(b/'logs/pipeline-complete.json'),cleanup=read(b/'logs/cleanup-status.json'),error_tails=errors)))
'''.replace('BASE',repr(base))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('deploy-data','observe-data','deploy-fit','observe-fit','stage-fit'));p.add_argument('--config',required=True)
    p.add_argument('--receipts',required=True);p.add_argument('--reuse-source-commit');p.add_argument('--reuse-source-repo');a=p.parse_args()
    c=json.loads(Path(a.config).read_text());repo=Path(__file__).resolve().parents[2]
    if a.phase.startswith('deploy-')or a.phase=='stage-fit':
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
        branch=subprocess.check_output(['git','branch','--show-current'],cwd=repo,text=True).strip()
        actual=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+branch],cwd=repo,text=True).split()[0]
        assert actual==commit
        script=deploy_code(c,commit,a.reuse_source_commit,a.reuse_source_repo,a.phase.split('-')[1],stage_only=a.phase=='stage-fit')
    else:script=(observation_code if a.phase=='observe-data'else fit_observation_code)(c['base'])
    tag=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    directory=Path(a.receipts);path=directory/('probe20k-'+a.phase+'-'+tag+'.transport.json')
    result=remote(script,str(path));values=[json.loads(line)for line in result.splitlines()if line.startswith('{')]
    output=path.with_name(path.stem.replace('.transport','')+'.json');output.write_text(json.dumps(values[-1],indent=2)+'\n')
    print(json.dumps(dict(actual=str(output),sha256=hashlib.sha256(output.read_bytes()).hexdigest(),result=values[-1])))
