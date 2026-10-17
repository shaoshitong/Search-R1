"""Code-only isolated deployment and once submission of the local fitting task."""
import argparse
import json
from pathlib import Path
import subprocess

from reproduction.prefix20_group_joint100.control import archive_for_commit
from reproduction.prefix_selection200.control import deployment_script
from reproduction.h1008_qprobe.established_controller_transport import remote


def build(config, commit, reuse_commit, reuse_repo):
    code = deployment_script(config['base'],commit,config,
                             archive_for_commit(commit,reuse_commit),reuse_commit,reuse_repo)
    # A pre-existing deployment returns instead of rerunning mutations. Recover
    # its authenticated receipt and continue only the still-unsubmitted scope.
    deployment = "exec(compile("+repr(code)+",'<code-only-local50-deploy>','exec'),{})"
    return r'''
import pathlib,json,hashlib,sys,glob,subprocess,os
b=pathlib.Path(BASE)
replacement=REPLACEMENT
if replacement:
 from_base=pathlib.Path(replacement['base']);job=str(replacement['job_id'])
 sys.path.insert(0,str(from_base/'repo'))
 from reproduction.common8.native_allocation import fields
 r=subprocess.run(['scontrol','show','job',job,'-o'],capture_output=True,text=True,timeout=30)
 assert r.returncode==0,(r.stdout,r.stderr)
 native=fields(r.stdout)
 assert native['Command']==str(from_base/'preparation/native-local-fit50.sh')
 if native['JobState']=='RUNNING':
  print(json.dumps(dict(status='ORIGINAL_LOCAL_JOB_ALREADY_RUNNING_NO_SECOND_SUBMISSION',job_id=job)));raise SystemExit
 receipt=from_base/'preparation/replaced-pending-for-authorized-sharing.json'
 if not receipt.exists():
  assert native['JobState']=='PENDING' and native['RunTime']=='00:00:00'
  stopped=subprocess.run(['scancel',job],capture_output=True,text=True,timeout=30)
  assert stopped.returncode==0,stopped.stderr
  with receipt.open('x')as stream:json.dump(dict(job_id=job,native_before=native,returncode=0,
   reason='authorized physical8 Slurm sharing; no actor/probe updates existed'),stream)
 else:
  assert native['JobState']=='CANCELLED' and json.loads(receipt.read_text())['job_id']==job
if not b.exists():
 DEPLOYMENT
else:
 assert (b/'preparation/deploy-receipt.json').exists(),'Reconcile incomplete deployment first'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
c=read(b/'preparation/launch-config.json');s=c['local_probe_fit50']
assert c['source_commit']==COMMIT and s['epochs']==10 and s['actor_outer']==50
proof=pathlib.Path(s['stop3434_proof']);assert sha(proof)==s['stop3434_proof_sha256']
t=read(proof);assert t['job_id']=='3434' and t['status']=='HUMAN_STOP_ORIGINAL_INODE_FULL_SHA_OWNED_RELEASED'
assert not t['actual_births_present'] and not t['native_job_cgroup_paths'] and not t['owned_gpu']
old=proof.parent.parent;sys.path.insert(0,str(b/'repo'))
from reproduction.prefix_only_correctness_ab.launch import birth_states
ledger=read(old/'logs/process-ledger.json');births=[v for job in ledger['jobs'].values()for v in job['ledger'].values()]
assert not birth_states(births) and not glob.glob('/sys/fs/cgroup/**/job_3434*',recursive=True)
gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,capture_output=True,timeout=20)
assert gpu.returncode==0 and not {str(v['pid'])for v in births}.intersection(gpu.stdout.split())
from reproduction.prefix20_online.launch import once
root=b/'outers/outer-000051';root.mkdir(parents=True,exist_ok=True)
q=read(pathlib.Path(s['source_questions']));q.update(outer=51,run_id=c['run_id'])
if not (root/'questions.json').exists():once(root/'questions.json',q)
else:assert read(root/'questions.json')==q
binding=dict(run_id=c['run_id'],outer=51,actor_version=c['actor_version'],
 actor_model_dir=c['actor_model_dir'],actor_model_files_sha256=c['actor_model_files_sha256'])
if not (root/'actor-binding.json').exists():once(root/'actor-binding.json',binding)
else:assert read(root/'actor-binding.json')==binding
from reproduction.prefix_selection200.local_probe_fit50 import submit
submitted=submit(b)
print(json.dumps(dict(status='LOCAL50_CODE_AND_FIXED_ACTOR_BOUND_SUBMISSION',base=str(b),
 source_commit=c['source_commit'],deploy=read(b/'preparation/deploy-receipt.json'),
 actor_version=c['actor_version'],questions=5120,diagnostic_questions=512,probe_epochs=10,
 learning_rate=s['learning_rate'],submit=submitted,old3434_second_release_check=True)))
'''.replace('BASE',repr(config['base'])).replace('REPLACEMENT',repr(config['local_probe_fit50'].get('replace_pending'))).replace('COMMIT',repr(commit)).replace('DEPLOYMENT',deployment)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--evidence',required=True)
    parser.add_argument('--reuse-commit',required=True);parser.add_argument('--reuse-repo',required=True);args=parser.parse_args()
    repo=Path(__file__).resolve().parents[2];commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=repo,text=True).strip()
    assert subprocess.check_output(['git','ls-remote','origin','refs/heads/'+branch],cwd=repo,text=True).split()[0]==commit
    value=remote(build(json.loads(Path(args.config).read_text()),commit,args.reuse_commit,args.reuse_repo),args.evidence)
    path=Path(args.evidence).with_suffix('.actual.json');path.write_text(value)
    print(json.dumps(dict(actual=str(path),result=json.loads(value.splitlines()[-1]))),flush=True)
