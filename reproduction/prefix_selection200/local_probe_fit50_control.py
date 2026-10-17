"""Code-only isolated deployment and once submission of the local fitting task."""
import argparse
import json
from pathlib import Path
import subprocess

from reproduction.prefix20_group_joint100.control import archive_for_commit
from reproduction.prefix_selection200.control import deployment_script
from reproduction.h1008_qprobe.established_controller_transport import remote


def terminal_script(base, job):
    return r'''
import pathlib,json,subprocess,sys,glob,hashlib,os,time
b=pathlib.Path(BASE);sys.path.insert(0,str(b/'repo'));job=JOB
from reproduction.prefix_only_correctness_ab.launch import birth_states
from reproduction.common8.native_allocation import fields
def read(p):return json.loads(p.read_text())
proof=b/'preparation'/('local50-'+job+'-native-terminal-original-inode-full-sha.json')
if proof.exists():print(json.dumps(dict(existing=True,path=str(proof),sha256=hashlib.sha256(proof.read_bytes()).hexdigest())));raise SystemExit
r=subprocess.run(['scontrol','show','job',job,'-o'],capture_output=True,text=True,timeout=30);assert r.returncode==0
native=fields(r.stdout);assert native['JobState']in('FAILED','CANCELLED','COMPLETED','OUT_OF_MEMORY','TIMEOUT')
ledger=read(b/'logs/process-ledger.json');births=[v for j in ledger['jobs'].values()for v in j['ledger'].values()]
present=birth_states(births);cgroups=glob.glob('/sys/fs/cgroup/**/job_'+job+'*',recursive=True)
gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=20)
owned=sorted(set(gpu.stdout.split()).intersection(str(v['pid'])for v in births));assert not present and not cgroups and not owned and gpu.returncode==0
logs=[]
for p in sorted(b.rglob('*.log')):
 if not p.is_file()or p.is_symlink():continue
 with p.open('rb')as stream:
  before=os.fstat(stream.fileno());h=hashlib.sha256()
  for chunk in iter(lambda:stream.read(8*1024**2),b''):h.update(chunk)
  after=os.fstat(stream.fileno())
 signature=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
 assert signature(before)==signature(after)==signature(p.stat())
 logs.append(dict(path=str(p),inode=before.st_ino,dev=before.st_dev,bytes=before.st_size,sha256=h.hexdigest()))
prefix=read(b/'outers/outer-000051/prefix/complete.json');assert prefix['trajectories']==25600
v=dict(status='LOCAL50_NATIVE_TERMINAL_ORIGINAL_LOG_SHA_OWNED_RELEASED',job_id=job,native=native,
 original_logs=logs,original_births=len(births),actual_births_present=present,native_job_cgroup_paths=cgroups,
 owned_gpu_pids=owned,prefix_complete_sha256=hashlib.sha256((b/'outers/outer-000051/prefix/complete.json').read_bytes()).hexdigest(),
 checked_unix=time.time(),probe_actual_new_adam=0)
with proof.open('x')as stream:json.dump(v,stream,sort_keys=True)
print(json.dumps(dict(path=str(proof),sha256=hashlib.sha256(proof.read_bytes()).hexdigest(),native=native['JobState'],original_logs=len(logs))))
'''.replace('BASE',repr(base)).replace('JOB',repr(str(job)))


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
if s.get('reuse_prefix'):
 reuse=s['reuse_prefix'];src=pathlib.Path(reuse['source_base']);proof=pathlib.Path(reuse['terminal_proof'])
 assert sha(proof)==reuse['terminal_proof_sha256']
 stopped=read(proof);assert stopped['job_id']==str(reuse['source_job_id'])and not stopped['actual_births_present']and not stopped['owned_gpu_pids']
 source_root=src/'outers/outer-000051';prefix=read(source_root/'prefix/complete.json')
 assert prefix['actor_version']==c['actor_version']and prefix['trajectories']==25600
 imported=b/'preparation/reused-complete-prefix.json'
 if not imported.exists():
  for source in sorted((source_root/'prefix').rglob('*')):
   assert not source.is_symlink()
   if source.is_file():
    target=root/'prefix'/source.relative_to(source_root/'prefix');target.parent.mkdir(parents=True,exist_ok=True);assert not target.exists();os.link(source,target)
  os.link(source_root/'all-selection.json',root/'all-selection.json')
  once(imported,dict(source_base=str(src),prefix_complete_sha256=sha(root/'prefix/complete.json'),
   all_selection_sha256=sha(root/'all-selection.json'),prefixes_regenerated=0,hidden_forward_repeated=False,source_originals_changed=False))
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
