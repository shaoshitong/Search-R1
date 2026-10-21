"""Authorized single existing-job walltime extension, no source/update change."""
import hashlib,importlib.util,json,pathlib,shlex,subprocess
ROOT=pathlib.Path(__file__).resolve().parents[4];CONTROL=ROOT.parents[1];OUT=pathlib.Path(__file__).parent
source='''import pathlib,json,hashlib,subprocess,time
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/direct-hidden-count30016-train-v2')
assert hashlib.sha256((b/'config.json').read_bytes()).hexdigest()=='3e910b12c0ff52eac70e61566c6135e2ec188bb47ec64ad79524abbd7a657526'
assert json.loads((b/'source-publication.json').read_text())['commit']=='501cf80478b77f68c3e13a96d4659e3da0971b2e'
def job():return subprocess.check_output(['scontrol','show','job','-o','743619']).decode()
before=job()
for token in ('JobId=743619 ','JobName=q7b-direct-count938 ','Account=mou176 ','QOS=normal ','JobState=RUNNING ','WorkDir='+str(b)+' '):assert token in before
assert 'TimeLimit=02:00:00 ' in before or 'TimeLimit=03:00:00 ' in before
if 'TimeLimit=03:00:00 ' in before:rc=0;stderr='';changed=False
else:
 r=subprocess.run(['scontrol','update','JobId=743619','TimeLimit=03:00:00'],stdout=subprocess.PIPE,stderr=subprocess.PIPE);rc=r.returncode;stderr=r.stderr.decode();changed=rc==0
after=job()
for token in ('JobId=743619 ','JobName=q7b-direct-count938 ','Account=mou176 ','QOS=normal ','JobState=RUNNING ','WorkDir='+str(b)+' '):assert token in after
if changed:assert 'TimeLimit=03:00:00 ' in after
result=dict(status='EXISTING_JOB_WALLTIME_EXTENDED_TO3H' if 'TimeLimit=03:00:00 ' in after else 'WALLTIME_EXTENSION_DENIED_OR_UNAVAILABLE_ORIGINAL_JOB_PRESERVED',job_id='743619',requested_TimeLimit='03:00:00',before=before,after=after,returncode=rc,stderr=stderr,job_changed=changed,normal_QOS_unchanged=True,source_and_config_unchanged=True,target_optimizer_updates=938,actor_updates=0,no_job_cancelled_or_submitted=True,time=time.time())
p=b/'walltime-extension-audit.json';assert not p.exists(),'Inspect existing extension receipt before repeating';p.write_text(json.dumps(result,indent=2)+'\\n');result['remote_audit']=dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest());print(json.dumps(result))'''
spec=importlib.util.spec_from_file_location('joint_walltime_control',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl)
command='python3 -c '+shlex.quote(source)
if len(command.encode())>=16000:raise ValueError('Control command exceeds16KB')
r=ctl.run(command,account='mou176',source_address=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0']).decode().strip(),timeout=45)
if r.returncode:raise RuntimeError('Outcome not confirmed; read existing job/receipt before any repeat: '+str(r.stderr[-1400:]))
result=json.loads(r.stdout);result['control_source_sha256']=hashlib.sha256(source.encode()).hexdigest()
(OUT/'actual-walltime-extension.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
