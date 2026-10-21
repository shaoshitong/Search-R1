"""Reconcile one already created held job after an unconfirmed SSH return.

Never invokes sbatch. Frozen command, source CP, account, resources, original
terminal state and exclusive locks must match before this same ID is released.
"""
import argparse,hashlib,importlib.util,json,pathlib,shlex,subprocess
ROOT=pathlib.Path(__file__).resolve().parents[4];CONTROL=ROOT.parents[1];OUT=pathlib.Path(__file__).parent
p=argparse.ArgumentParser();p.add_argument('--job-id',required=True);args=p.parse_args()
if not args.job_id.isdigit() or args.job_id=='743619':p.error('Actual unique held continuation ID required')
code=r'''import pathlib,json,subprocess,hashlib,fcntl,os,time
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/direct-hidden-count30016-train-v2');a=b/'continuations/attempt-0001-v2'
def read(p):return json.loads(pathlib.Path(p).read_text())
def command(v):return subprocess.check_output(v).decode()
def ref(p):
 p=pathlib.Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def immutable(p,v):
 if p.exists():assert read(p)==v
 else:
  tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,sort_keys=True));tmp.replace(p)
lock=(a/'operator-control.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
training=(b/'joint-training.lock').open('a+');fcntl.flock(training,fcntl.LOCK_EX|fcntl.LOCK_NB)
prepared=read(a/'preparation.json');intent=read(a/'submit-intent.json');pub=read(a/'operator-publication.json')
assert pub['commit']=='31e930ce8327078612c3ee9dd98aec2188a78826' and intent['operator_commit']==pub['commit']
assert prepared['status']=='ACTUAL_INTERRUPTED_JOINT_STATE_VERIFIED_CONTINUATION_PREPARED_NOT_SUBMITTED'
assert prepared['source_cursor']==770 and prepared['remaining_updates']==168
assert read(b/'latest.json')['sha256']==prepared['source_checkpoint']['sha256']==intent['source_checkpoint']['sha256']
assert read(b/'progress.json')['cursor']==770 and read(b/'update-intent.json')['next_cursor']<=770 and read(b/'interrupted.json')['cursor']==770
assert ref(b/'config.json')['sha256']=='3e910b12c0ff52eac70e61566c6135e2ec188bb47ec64ad79524abbd7a657526'
assert ref(prepared['sbatch']['path'])==prepared['sbatch']==intent['sbatch']
assert not (a/'failure.json').exists() and not (b/'failure.json').exists()
assert not command(['squeue','-j','743619','-h','-o','%i|%T']).strip()
terminal=command(['sacct','-j','743619','-X','-n','-P','--format=JobIDRaw,State,ExitCode']);assert '743619|COMPLETED|0:0' in terminal
name='q7b-direct-count938-r1';rows=command(['squeue','-u','mou176','-h','-o','%i|%j|%T']).splitlines();same=[r.split('|')[0] for r in rows if len(r.split('|'))>1 and r.split('|')[1]==name];assert same==[JID]
identity=command(['scontrol','show','job','-o',JID]);fields=dict(t.split('=',1) for t in identity.split() if '=' in t)
expected=dict(JobId=JID,JobName=name,Account='mou176',UserId='mou176(%d)'%os.getuid(),QOS='normal',Partition='acd_u',JobState='PENDING',Reason='JobHeldUser',TimeLimit='03:00:00',NumCPUs='8',NumTasks='1',Dependency='(null)',Command=prepared['sbatch']['path'],WorkDir=str(b),MinMemoryNode='64G',TresPerNode='gres/gpu:1')
assert all(fields.get(k)==v for k,v in expected.items()),'Actual held job identity/resources differ; never create another job'
if (a/'job.id').exists():assert (a/'job.id').read_text().strip()==JID
else:(a/'job.id').write_text(JID+'\n')
submission=dict(job_id=JID,identity=identity,source_checkpoint=prepared['source_checkpoint'],source_cursor=770,remaining_updates=168,dependency=None,operator_commit=pub['commit'],runtime_commit=prepared['runtime_commit'],time=time.time(),reconciled_from_unconfirmed_submit=True)
if (a/'submission.json').exists():assert read(a/'submission.json')['job_id']==JID
else:immutable(a/'submission.json',submission)
proof=dict(status='UNIQUE_ALREADY_CREATED_HELD_RESUME_IDENTITY_RECONCILED_NO_SBATCH',job_id=JID,source_cursor=770,source_checkpoint=prepared['source_checkpoint'],sbatch=prepared['sbatch'],identity=identity,original_terminal=terminal,source_control_sha256=CONTROL_SHA,time=time.time(),duplicate_submissions=0)
immutable(a/'submission-reconciliation.json',proof)
command(['scontrol','release',JID]);result=dict(status='RECONCILED_EXISTING_CONTINUATION_RELEASED_NO_DUPLICATE',job_id=JID,source_cursor=770,remaining_updates=168,source_checkpoint_sha256=prepared['source_checkpoint']['sha256'],source_control_sha256=CONTROL_SHA,actual_submission_count=1,time=time.time());immutable(a/'submission-reconciliation-release.json',result);print(json.dumps(result))
'''
spec=importlib.util.spec_from_file_location('joint_held_resume_reconciliation',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl)
script='JID='+repr(args.job_id)+'\nCONTROL_SHA='+repr(hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest())+'\n'+code
r=ctl.run('python3 -c '+shlex.quote(script),account='mou176',source_address=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0']).decode().strip(),timeout=45)
if r.returncode:raise RuntimeError('Reconciliation outcome unconfirmed; inspect same job and intent, do not submit another: '+str(r.stderr[-1800:]))
value=json.loads(r.stdout);(OUT/'actual-resume-reconciled-release.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))
