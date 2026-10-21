"""One guarded own-process USR1 for an exact saved resource handoff.

Preparation is safe at any cursor. Preferred --execute follows actual768
fixed evaluation with5..20min remaining. At5..10min, any accepted cursor>=256
may safely hand off once its due evaluations are complete;768 is not required.
This does
not cancel a job, change its source, reset state or submit a continuation.
"""
import argparse,importlib.util,json,pathlib,shlex,subprocess
ROOT=pathlib.Path(__file__).resolve().parents[4];CONTROL=ROOT.parents[1];OUT=pathlib.Path(__file__).parent
p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');args=p.parse_args()
code=r'''import pathlib,json,os,signal,fcntl,hashlib,subprocess,time
base=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/direct-hidden-count30016-train-v2');job='743619';config='3e910b12c0ff52eac70e61566c6135e2ec188bb47ec64ad79524abbd7a657526'
def read(p):return json.loads(pathlib.Path(p).read_text())
def atomic(p,v):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,sort_keys=True));tmp.replace(p)
def seconds(v):
 days=0
 if '-' in v:d,v=v.split('-');days=int(d)
 t=[int(x) for x in v.split(':')]
 if len(t)==2:t=[0]+t
 if len(t)!=3:raise ValueError('Unrecognized finite original job TimeLeft')
 return days*86400+t[0]*3600+t[1]*60+t[2]
assert (base/'job.id').read_text().strip()==job
assert hashlib.sha256((base/'config.json').read_bytes()).hexdigest()==config
pub=read(base/'source-publication.json');assert pub['commit']=='501cf80478b77f68c3e13a96d4659e3da0971b2e'
queue=subprocess.check_output(['squeue','-j',job,'-h','-o','%L|%T|%j|%a']).decode().strip().split('|')
assert len(queue)==4 and queue[1:]==['RUNNING','q7b-direct-count938','mou176']
remaining=seconds(queue[0]);progress=read(base/'progress.json');latest=read(base/'latest.json');controller=read(base/'controller.json');cursor=progress['cursor']
assert progress['steps']==10464+cursor and progress['head_steps']==cursor and latest['cursor']<=cursor
result=dict(status='OWN_JOB_HANDOFF_WINDOW_OBSERVED_NO_SIGNAL',job_id=job,cursor=cursor,latest_cursor=latest['cursor'],seconds_remaining=remaining,time=time.time(),signal_sent=False)
if not EXECUTE:print(json.dumps(result));raise SystemExit(0)
preferred=768<=cursor<938 and 300<=remaining<=1200
fallback=256<=cursor<938 and 300<=remaining<=600
assert preferred or fallback,'Require preferred768/5..20min or any accepted cursor/5..10min safety fallback'
result['handoff_mode']='preferred_after_768' if preferred else 'near_deadline_any_accepted_cursor'
assert not any((base/n).exists() for n in ('failure.json','first-cp-audit-failure.json','interrupted.json','complete.json'))
for milestone in (256,512,768):
 if milestone>cursor:continue
 evaluation=read(base/'evaluations'/('cursor-%06d'%milestone)/'complete.json');assert evaluation['cursor']==milestone and evaluation['config_sha256']==config and evaluation['training_nine_components_unchanged'] is True
assert latest['full_CPU_reload_passed'] is True and latest['full_file_SHA256_verified'] is True
checkpoint=pathlib.Path(latest['path']);assert checkpoint.is_file() and checkpoint.stat().st_size==latest['bytes']
intent=read(base/'update-intent.json');assert intent['next_cursor']<=cursor+1
assert controller['job_id']==job and controller['status']=='JOINT_E1_INPUTS_ACCEPTED'
pid=controller['pid'];process=pathlib.Path('/proc')/str(pid);assert process.stat().st_uid==os.getuid()
command=(process/'cmdline').read_bytes();env=(process/'environ').read_bytes()
assert b'reproduction.operations.qwen7b_warmup.direct_hidden_count_train\x00--action\x00train\x00' in command
assert b'--config-sha256\x00'+config.encode()+b'\x00' in command
assert b'SLURM_JOB_ID='+job.encode()+b'\x00' in env
assert b'QWEN7B_CODE_ROOT='+pub['code_root'].encode()+b'\x00' in env
work=base/'continuations/attempt-0001-v2';assert read(work/'CPU-qualification.json')['tests']==4
lock=(work/'safe-signal.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
path=work/'safe-signal-intent.json'
assert not path.exists(),'Unknown or completed signal intent: inspect existing proof, do not repeat'
result.update(status='OWN_PROCESS_USR1_INTENT_FOR_HIGHEST_SAVED_JOINT_HANDOFF',pid=pid,latest_checkpoint_sha256=latest['sha256'],highest_update_intent=intent,original_runtime_commit=pub['commit'],partial_rollback_permitted=False)
atomic(path,result)
os.kill(pid,signal.SIGUSR1)
result.update(status='OWN_PROCESS_USR1_SENT_AWAIT_ACTUAL_INTERRUPTED_AND_TERMINAL',signal_sent=True,time=time.time())
atomic(work/'safe-signal-receipt.json',result);print(json.dumps(result))
'''
spec=importlib.util.spec_from_file_location('joint_exact_handoff_control',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl)
command='srun --jobid=743619 --overlap --gres=gpu:0 --ntasks=1 --cpus-per-task=1 python3 -c '+shlex.quote('EXECUTE='+repr(args.execute)+'\n'+code)
result=ctl.run(command,account='mou176',source_address=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0']).decode().strip(),timeout=40)
if result.returncode:raise RuntimeError('Guarded signal outcome not accepted; inspect state and own signal intent before retry: '+result.stderr[-1800:])
value=json.loads(result.stdout);(OUT/('actual-safe-handoff-signal.json' if args.execute else 'actual-safe-handoff-dryrun.json')).write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))
