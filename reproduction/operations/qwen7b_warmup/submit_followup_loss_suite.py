"""Explicitly submit each independently prepared matched followup once."""
import argparse,importlib.util,json,shlex,subprocess
from pathlib import Path

REMOTE=r'''
import pathlib,json,subprocess,hashlib,fcntl,time
B=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019');suite=B/'algorithm-improvement-20261021/matched-loss128-followup-v1'/group_base
read=lambda p:json.loads(pathlib.Path(p).read_text())
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def command(args):
 p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=25);assert p.returncode==0,p.stdout+p.stderr;return p.stdout.strip()
def atomic(p,v):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)
with (suite/'submission.lock').open('a') as guard:
 fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 prepared=read(suite/'preparation.json');pub=read(suite/'source-publication.json');assert prepared['group_base']==group_base and prepared['start_from_sealed_E1'] and not prepared['continues128step_result']
 for name,digest in pub['files'].items():assert sha(pathlib.Path(pub['code_root'])/name)==digest
 # Record the actual screening comparison supporting the selected candidate base.
 baselines={}
 for name in ['trajectory','joint','event']:
  p=B/'algorithm-improvement-20261021/matched-loss128-v2'/name/'complete.json';c=read(p);assert c['cursor']==128 and c['steps']==10592 and c['status']=='MATCHED_PROBE_LOSS_ABLATION_COMPLETE';m=c['metrics']['selection2048'];baselines[name]=dict(mixed_ba=m['pool']['mixed_balanced_accuracy'],auc=m['pool']['mixed_roc_auc'],mixed_at512=m['selected512']['mixed_questions'],receipt_sha256=sha(p))
 assert all(baselines[group_base][k]>=v[k] for v in baselines.values() for k in ['mixed_ba','auc','mixed_at512']),'Selected group base must be supported by this matched screen'
 choice=dict(group_base=group_base,screening_comparison=baselines,final_success_claimed=False,budget=128,source_E1_preserved=True,time=time.time())
 if not (suite/'base-choice.json').exists():atomic(suite/'base-choice.json',choice)
 records=[]
 for name,item in prepared['variants'].items():
  base=pathlib.Path(item['base']);job_name=item['job_name']
  for key in ['config','sbatch']:assert sha(item[key]['path'])==item[key]['sha256']
  if (base/'job.id').exists():
   j=(base/'job.id').read_text().strip();records.append(dict(variant=name,job_id=j,status='EXISTING_JOB_NOT_RESUBMITTED',accounting=command(['sacct','-X','-n','-P','-j',j,'-o','JobID,State,ExitCode'])));continue
  existing=command(['squeue','-h','-u','mou176','-n',job_name,'-o','%i|%T|%R'])
  if existing:
   assert len(existing.splitlines())==1 and 'JobHeldUser' in existing,'Unknown same-name job; inspect before proceeding';j=existing.split('|')[0]
   state=command(['scontrol','show','job',j,'-o']);assert 'UserId=mou176(' in state and 'Command='+str(base/'fit.sbatch')+' ' in state
  else:
   assert not (base/'submit-intent.json').exists(),'Unknown previous submission outcome; reconcile rather than duplicate'
   atomic(base/'submit-intent.json',dict(job_name=job_name,config=item['config'],source_commit=pub['commit'],driver_commit=driver_commit,time=time.time()))
   j=command(['sbatch','--hold','--parsable','--time=01:00:00',str(base/'fit.sbatch')]).split(';')[0];assert j.isdigit()
  with (base/'job.id').open('x') as f:f.write(j+'\n')
  rec=dict(variant=name,job_id=j,base=str(base),config=item['config'],source_commit=pub['commit'],driver_commit=driver_commit,gpus=1,partition='acd_u',no_dependency=True,status='HELD_IDENTITY_VERIFIED',time=time.time())
  atomic(base/'submission.json',rec);s=command(['squeue','-h','-j',j,'-o','%T|%R']);assert s.startswith('PENDING|') and 'JobHeldUser' in s
  command(['scontrol','release',j]);rec.update(status='RELEASED_VERIFY_ACTUAL_UPDATES',queue=command(['squeue','-h','-j',j,'-o','%i|%j|%T|%P|%R']));atomic(base/'submission.json',rec);records.append(rec)
 out=dict(status='FOLLOWUP_SUBMISSIONS_RECONCILED_NOT_TRAINING_ACCEPTANCE',group_base=group_base,source_commit=pub['commit'],records=records,time=time.time());atomic(suite/'submissions.json',out);print(json.dumps(out))
'''

def main():
 p=argparse.ArgumentParser();p.add_argument('--control-root',type=Path,required=True);p.add_argument('--group-base',choices=['event','trajectory','joint'],required=True);a=p.parse_args();repo=Path(__file__).resolve().parents[3]
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip();assert subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/qwen7b-probe-warmup-20261019'],cwd=repo,text=True).split()[0]==commit
 assert Path(__file__).read_bytes()==subprocess.check_output(['git','show',commit+':'+str(Path(__file__).relative_to(repo))],cwd=repo)
 spec=importlib.util.spec_from_file_location('control',a.control_root/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);address=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip();code='group_base='+repr(a.group_base)+'\ndriver_commit='+repr(commit)+'\n'+REMOTE
 r=m.run('python3 -c '+shlex.quote(code),account='mou176',source_address=address,timeout=90)
 values=[json.loads(x) for x in r.stdout.decode().splitlines() if x.startswith('{')]
 if r.returncode or not values:raise RuntimeError('Submission outcome requires inspection; do not repeat blindly: '+r.stderr.decode()[-1500:]+' '+r.stdout.decode()[-1000:])
 (a.control_root/'outputs/qwen7b-warmup-20261019'/('followup-'+a.group_base+'-submissions.json')).write_text(json.dumps(values[-1],indent=2));print(json.dumps(values[-1]))
if __name__=='__main__':main()
