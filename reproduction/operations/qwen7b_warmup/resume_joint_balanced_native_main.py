"""Terminal-only exact recovery controller for the accepted124194 main namespace.

This never alters the bound config or active source. A live/pending same-name
writer returns WAIT before checkpoint hashing or any sbatch. The original
24-hour allocation must exit with its highest accepted transaction saved.
"""
import argparse,base64,hashlib,importlib.util,json,shlex,subprocess,zlib
from pathlib import Path
R=Path(__file__).resolve().parents[3];CONTROL=R.parents[1]
OUT=R/'reproduction/reports/qwen7b-algorithm-improvement-20261021/joint-balanced-native-main'
BODY='''import pathlib,json,hashlib,subprocess,time,os,fcntl,re
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/joint-balanced-native-E4-124194-v1');original='745501';recovery=b/'recovery-attempt-001';code=b/'code-3b56e583efe2'
def read(p):return json.loads(pathlib.Path(p).read_bytes())
def proof(p):
 p=pathlib.Path(p);h=hashlib.sha256()
 with p.open('rb')as f:
  for raw in iter(lambda:f.read(2*1024*1024),b''):h.update(raw)
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def run(args):
 r=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=15);assert r.returncode==0,r.stderr[-1200:];return r.stdout.strip()
def immutable(p,value):
 raw=(json.dumps(value,sort_keys=True,indent=2)+chr(10)).encode()
 if p.exists():assert p.read_bytes()==raw,'Existing recovery metadata differs: '+str(p)
 else:
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
def active():
 rows=run(['squeue','-h','-u','mou176','-o','%i|%j|%T']).splitlines();return[x for x in rows if x.split('|')[1]in('q7b-native-joint124194','q7b-native-joint124194-resume')]
assert read(b/'submission-result.json')['job_id']==original
assert proof(b/'config.json')['sha256']=='5aaa63c9348f410357be92ce491ba6d18286d618ab86dae6e63f193f2b4367b6'
assert proof(b/'training.sbatch')['sha256']=='073299d9f26bf858e6d61fbc8c12d6b6d726079385ff010796bb3e89a84e1d51'
assert proof(b/'source-publication.json')['sha256']=='e1a25104cc649d66ae166e56b7c43dc0dc060ee9e3f73d6ba1b2f2a09254a293'
writer=active();out=dict(epoch=time.time(),action=v['action'],namespace=str(b),active_same_main_jobs=writer,config_unchanged=True,GPU_submitted=False)
if writer:
 out.update(status='WAITING_FOR_ORIGINAL_WRITER_TERMINAL_AND_EXACT_HIGHEST_CHECKPOINT',checkpoint_tensor_loaded=False)
 if(b/'progress.json').exists():out['actual_cursor']=read(b/'progress.json')['cursor']
 if(b/'latest.json').exists():out['latest_complete_cursor']=read(b/'latest.json')['cursor']
 print(json.dumps(out));raise SystemExit(0)
rows=run(['sacct','-n','-P','-j',original,'--format=JobIDRaw,State,ExitCode,Elapsed,NodeList']).splitlines();main=[r for r in rows if r.split('|')[0]==original];assert len(main)==1
assert main[0].split('|')[1].split()[0].rstrip('+')in('COMPLETED','TIMEOUT','CANCELLED','PREEMPTED','FAILED'),'Exact original main terminal required'
if(b/'complete.json').exists():
 assert read(b/'complete.json')['cursor']==15528;out.update(status='ORIGINAL_FOUR_EPOCH_GOAL_ALREADY_COMPLETE_NO_RECOVERY');print(json.dumps(out));raise SystemExit(0)
assert not(b/'failure.json').exists(),'Actual training failure requires explicit repair/binding migration, not blind same-source resume'
lock=(b/'training.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
ctl=(b/'recovery-control.lock').open('a+');fcntl.flock(ctl,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not active(),'Writer appeared during recovery gate'
c=read(b/'config.json');latest=read(b/'latest.json');p=read(b/'progress.json');intent=read(b/'update-intent.json');stop=read(b/'interrupted.json');journal=read(b/'native-checkpoint-transaction.json');cursor=latest['cursor']
assert 1<=cursor<15528 and p['cursor']==intent['cursor']==stop['cursor']==cursor,'No unsaved accepted or unresolved later transaction may be rolled back'
assert intent['phase']=='accepted_update_complete'and journal['phase']=='published'and journal['new_checkpoint']==latest
assert stop['checkpoint']==latest and p['steps']==latest['steps']==stop['steps']==14418+cursor
assert latest['config_sha256']==proof(b/'config.json')['sha256']and latest['matched_binding_sha256']=='01071ad024af352d576a2d02f7e8d4a867f71bd7f6609ca9110db13e9b2f307f'
assert latest['full_CPU_six_state_reload_passed']and latest['sampler_full_state_and_cursor_reload_passed']and latest['saved_all_four_source_RNG_states']
assert latest['sampler_state_sha256']==stop['sampler_state_sha256']==latest['data_cursor']['sampler_state_sha256']and latest['data_cursor']['cursor']==cursor and latest['data_cursor']['sampler_binding']=='c9281ec1c271c5d6db71a8d77789a436619e4d707f1a8426393638119e9cc45d'
assert set(latest['accepted_hashes'])==set(('raw','ema','adam','raw_buffers','ema_buffers','rng'))and all(re.fullmatch('[0-9a-f]{64}',s)for s in latest['accepted_hashes'].values())
cp=pathlib.Path(latest['path']);assert cp.parent==b/'checkpoints'and cp.is_file()and not cp.is_symlink();assert proof(cp)=={k:latest[k]for k in('path','bytes','sha256')},'Highest complete checkpoint fresh full SHA differs'
for ep in range(cursor//3882):assert(b/'evaluations'/('epoch-%02d'%ep)/'complete.json').is_file(),'Past required epoch evaluation missing'
pending_ref=None;pending=b/'pending-evaluation.json'
if pending.exists():
 pe=read(pending)
 # An old completed pending marker may remain; only an unevaluated current epoch is due.
 if not(b/'evaluations'/('epoch-%02d'%pe['epoch'])/'complete.json').exists():
  assert pe['cursor']==cursor and cursor%3882==0 and pe['epoch']==cursor//3882 and pe['checkpoint']==latest,'Pending epoch phase must match exact current checkpoint'
  pending_ref=proof(pending)
resume_phase='fixed_epoch_evaluation_due_before_next_update'if cursor%3882==0 and not(b/'evaluations'/('epoch-%02d'%(cursor//3882))/'complete.json').exists()else'next_native_optimizer_update'
recovery.mkdir(exist_ok=True);entry=(b/'training.sbatch').read_text().replace('--job-name=q7b-native-joint124194','--job-name=q7b-native-joint124194-resume').replace('--time=1-00:00:00','--time=2-00:00:00')
script=recovery/'training.sbatch'
if script.exists():assert script.read_text()==entry
else:script.write_text(entry)
subprocess.check_call(['bash','-n',str(script)])
ready=dict(status='EXACT_HIGHEST_NATIVE_MAIN_CHECKPOINT_RECOVERY_READY',epoch=time.time(),resume_phase=resume_phase,pending_evaluation=pending_ref,all_preceding_epoch_evaluations_preserved=True,original_main_terminal=main[0],checkpoint=latest,config=proof(b/'config.json'),publication=proof(b/'source-publication.json'),original_entry=proof(b/'training.sbatch'),recovery_entry=proof(script),interrupted=proof(b/'interrupted.json'),intent=proof(b/'update-intent.json'),journal=proof(b/'native-checkpoint-transaction.json'),runtime_source_commit='3b56e583efe2469ac9108873c31de7144a5201e6',current_world_size=1,remaining_updates=15528-cursor,old_Adam_steps=14418+cursor,source_E4_restart=False,checkpoint_tensor_load_deferred_to_owned_recovery_allocation=True,exact_native_and_sampler_next_update_double_replay_required=True,controller=v['controller'])
# Epoch time must not make a repeated unchanged prepare mutate an existing receipt.
if(recovery/'preparation.json').exists():
 old=read(recovery/'preparation.json');ready['epoch']=old['epoch'];assert old==ready
immutable(recovery/'preparation.json',ready);out.update(ready)
if v['action']=='submit':
 assert not(recovery/'submission-intent.json').exists(),'Unknown prior submission: reconcile same-name job, never repeat sbatch'
 assert not active();immutable(recovery/'submission-intent.json',dict(epoch=time.time(),ready=ready,only_namespace_continuation=True))
 r=subprocess.run(['sbatch','--hold','--parsable','--account=mou176',str(script)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=25)
 if r.returncode or not re.fullmatch(r'[0-9]+(?:;[^ ]+)?',r.stdout.strip()):raise RuntimeError('Uncertain submit; retained intent, inspect same-name job: '+r.stdout[-200:]+r.stderr[-800:])
 job=r.stdout.strip().split(';')[0];raw=run(['scontrol','show','job','-o',job])
 for item in('JobName=q7b-native-joint124194-resume','Account=mou176','QOS=normal','Partition=acd_u','TimeLimit=2-00:00:00','NumCPUs=8','MinMemoryNode=128G','Dependency=(null)','Command='+str(script),'WorkDir='+str(b)):assert item in raw,item
 assert 'gres/gpu=1'in raw or'TresPerNode=gres/gpu:1'in raw
 identity=dict(job_id=job,job=raw,checkpoint=latest,preparation=proof(recovery/'preparation.json'),epoch=time.time());immutable(recovery/'submission-identity.json',identity)
 # The new job is held and fully bound; free the writer lock before its release.
 fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
 run(['scontrol','release',job]);out.update(status='EXACT_NATIVE_MAIN_RECOVERY_JOB_RELEASED',job_id=job,GPU_submitted=True)
 immutable(recovery/'submission-result.json',out)
print(json.dumps(out))
'''
def main():
 p=argparse.ArgumentParser();p.add_argument('--action',choices=('prepare','submit'),required=True);a=p.parse_args();own=Path(__file__);commit=subprocess.check_output(['git','log','-1','--format=%H','--',str(own.relative_to(R))],cwd=R).decode().strip();remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/qwen7b-probe-warmup-20261019'],cwd=R).decode().split()[0];subprocess.check_call(['git','merge-base','--is-ancestor',commit,remote],cwd=R);assert own.read_bytes()==subprocess.check_output(['git','show',commit+':'+str(own.relative_to(R))],cwd=R)
 compile(BODY,'native-exact-recovery-remote','exec');spec=importlib.util.spec_from_file_location('native_exact_recovery_control',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl);ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0']).decode().strip();body='v='+repr(dict(action=a.action,controller=dict(commit=commit,path=str(own.relative_to(R)),sha256=hashlib.sha256(own.read_bytes()).hexdigest())))+'\n'+BODY;packed=base64.b64encode(zlib.compress(body.encode())).decode();cmd='python3 -c '+shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(packed)+')))');assert len(cmd.encode())<16000;r=ctl.run(cmd,account='mou176',source_address=ip,timeout=55)
 if r.returncode:raise RuntimeError(r.stdout[-2500:]+r.stderr[-1500:])
 out=json.loads(r.stdout.splitlines()[-1]);OUT.mkdir(parents=True,exist_ok=True);(OUT/('actual-exact-recovery-'+a.action+'.json')).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
