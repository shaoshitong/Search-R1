"""Compact readonly owned-fast-run evidence; never tensor load, signal or submit."""
import base64,hashlib,importlib.util,json,shlex,subprocess,zlib
from pathlib import Path
R=Path(__file__).resolve().parents[3];CONTROL=R.parents[1];REL='reproduction/operations/qwen7b_warmup/observe_mixed_direct_fast.py';BRANCH='codex/mixed-direct-fast-continuation-20261022';OUT=R/'reproduction/reports/qwen7b-algorithm-improvement-20261021/mixed-direct-fast-continuation'
BODY='''import pathlib,json,hashlib,subprocess,time,os
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/joint-mixed-direct-official-169028-fast-v1');job='746220'
def read(p):
 p=pathlib.Path(p);assert p.stat().st_size<16*1024**2 and p.suffix!='.pt';return json.loads(p.read_bytes())
def proof(p):
 p=pathlib.Path(p);assert p.stat().st_size<16*1024**2 and p.suffix!='.pt';raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def optional(name):
 for attempt in range(3):
  try:return read(b/name)
  except FileNotFoundError:
   if attempt<2:time.sleep(.03)
 return None
def compact(x,keys):return{k:x[k]for k in keys if k in x}
submit=read(b/'submission-result.json');assert submit['job_id']==job;ready=read(b/'preparation.json');cfg=read(b/'config.json');assert proof(b/'config.json')==ready['config'];pub=read(b/'source-publication.json');assert proof(b/'source-publication.json')==ready['publication'];drift=[]
for name,item in pub['files'].items():
 if proof(pathlib.Path(pub['code_root'])/name)!=dict(path=str(pathlib.Path(pub['code_root'])/name),**item):drift.append(name)
assert not drift,'Published new source drift: '+repr(drift)
hand=read(b/'source-handoff.json');spec=cfg['continuation'];assert hand['cursor']==ready['source_cursor']==373 and spec['source_checkpoint']==hand['pinned_checkpoint']and spec['source_checkpoint_original']==hand['checkpoint']and spec['source_checkpoint']['accepted_hashes']==hand['accepted_hashes'];s=pathlib.Path(spec['source_checkpoint']['path']).stat();assert s.st_size==spec['source_checkpoint']['bytes']and s.st_ino==hand['pinned_stat']['inode']and s.st_uid==os.getuid();r=subprocess.run(['sacct','-X','-n','-P','-j',job,'--format=JobIDRaw,State,ExitCode,Elapsed,NodeList'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=15);assert r.returncode==0
out=dict(epoch=time.time(),job_id=job,main_sacct=r.stdout.strip(),config=proof(b/'config.json'),source_publication=proof(b/'source-publication.json'),published_new_files_checked=len(pub['files']),source_drift=drift,source_cursor=373,source_checkpoint_SHA_from_inherited_verified_receipt=spec['source_checkpoint']['sha256'],source_9_hashes_bound=True,no_checkpoint_payload_read=True,micro_questions=cfg['architecture']['micro_questions'],prefetch_batches=cfg['prefetch_batches'],parallel_workers=cfg['parallel_workers'])
for name,keys in [('controller.json',('pid','job_id','time','stage','config_sha256','source_cursor')),('progress.json',('cursor','source_cursor','new_updates','time','seconds','native_steps','head_steps','binary_Mixed_BCE','native_gradient_L2','group_head_gradient_L2','micro_questions','prefetch','reader_integrity_mode')),('latest.json',('cursor','native_steps','head_steps','bytes','sha256','full_CPU_native_head_two_Adam_RNG_sampler_source_reload_passed')),('startup-qualification.json',('status','source_cursor','cursor','new_complete_updates','micro_questions','asynchronous_CPU_producer','global_clocks_not_reset'))]:
 x=optional(name)
 if x is not None:out[name[:-5]]=dict(compact(x,keys),receipt=proof(b/name))
x=optional('startup-binding-migration.json')
if x is not None:
 assert x['source_accepted_hashes']==x['new_accepted_hashes']==spec['source_checkpoint']['accepted_hashes'];out['migration']=dict(status=x['status'],source_cursor=x['source_cursor'],all9_hashes_preserved=True,full_checkpoint_fresh_SHA_and_nine_state_restore_passed=x['restoration_receipt']['full_checkpoint_fresh_SHA_and_nine_state_restore_passed'],global_Adam_sampler_RNG_preserved=x['global_Adam_sampler_RNG_preserved'],receipt=proof(b/'startup-binding-migration.json'))
for name in('first-fast-update-replay.json','fast-durable-next-update-replay.json'):
 x=optional(name)
 if x is not None:
  assert x['status']=='PASSED'and x['expected']==x['actual'];out[name[:-5]]=dict(compact(x,('status','source_cursor','cursor','native_changed','group_head_changed','micro_questions')),all9_state_hashes_exact=True,receipt=proof(b/name))
x=optional('first-fast-cp-independent-audit.json')
if x is not None:out['first-fast-cp-independent-audit']=dict(compact(x,('status','source_cursor','cursor','native_steps','head_steps','next_sampler_batch_identity_exact','global_clocks_not_reset','GPU','host','job_id')),receipt=proof(b/'first-fast-cp-independent-audit.json'))
x=optional('reader-integrity.json')
if x is not None:out['reader-integrity']=dict(compact(x,('mode','whole_payload_fresh_SHA_claimed','per_used_row_SHA_required','disposable_CPU_queue_not_checkpoint_state')),receipt=proof(b/'reader-integrity.json'))
x=optional('failure.json')
if x is not None:out['failure']=dict(type=x.get('type'),message=x.get('message'),traceback_tail=x.get('traceback','')[-5000:],time=x.get('time'))
log=b/'first-fast-cp-independent-audit.log'
if log.exists():
 out['independent_CPU_log']=dict(bytes=log.stat().st_size,mtime=log.stat().st_mtime)
 if out.get('failure'):
  with log.open('rb')as f:f.seek(max(0,log.stat().st_size-3000));out['independent_CPU_log']['tail']=f.read().decode(errors='replace')
print(json.dumps(out,sort_keys=True))
'''
def main():
 commit=subprocess.check_output(['git','log','-1','--format=%H','refs/heads/'+BRANCH,'--',REL],cwd=R).decode().strip();own=Path(__file__);assert own.read_bytes()==subprocess.check_output(['git','show',commit+':'+REL],cwd=R);remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+BRANCH],cwd=R).decode().split()[0];subprocess.check_call(['git','merge-base','--is-ancestor',commit,remote],cwd=R);compile(BODY,'compact_fast_observer','exec');spec=importlib.util.spec_from_file_location('private_fast_observation',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl);ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0']).decode().strip();packed=base64.b64encode(zlib.compress(BODY.encode())).decode();command='python3 -c '+shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(packed)+')))');assert len(command.encode())<16000;r=ctl.run(command,account='mou176',source_address=ip,timeout=45)
 if r.returncode:raise RuntimeError(r.stdout[-2000:]+r.stderr[-3000:])
 result=json.loads(r.stdout.splitlines()[-1]);OUT.mkdir(parents=True,exist_ok=True);(OUT/'actual-compact-observation.json').write_text(json.dumps(result,indent=2)+'\n')
 with(OUT/'actual-compact-observations.jsonl').open('a')as f:f.write(json.dumps(result,sort_keys=True)+'\n')
 print(json.dumps(result))
if __name__=='__main__':main()
