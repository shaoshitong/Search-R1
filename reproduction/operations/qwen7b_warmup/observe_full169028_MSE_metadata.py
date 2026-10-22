"""Durable, bounded, read-only metadata observer for two exact training jobs.

Never reads checkpoints/tensors/final pools, mutates Slurm or training files.
Only this observer's independent directory is writable.
"""
import argparse,collections,fcntl,hashlib,json,math,os,subprocess,time,traceback
from pathlib import Path

def read(p):return json.loads(Path(p).read_bytes())
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':')).encode()
def atomic(p,x):
 p=Path(p);tmp=p.with_name(p.name+'.%d.tmp'%os.getpid());tmp.write_bytes(canonical(x)+b'\n');tmp.replace(p)
def proof(p):
 p=Path(p);raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def checked(ref):
 p=Path(ref['path']);assert proof(p)=={k:ref[k]for k in('path','bytes','sha256')};return p
KEYS=('stage','epoch','cursor','batch_attempts','actual_Adam_updates','EMA_updates','Adam_and_EMA_step_performed','rank','native_gradient_L2','head_gradient_L2','training_loss','actual_global_batch','local_batch','gradient_accumulation','micro','prefill_rows_equal_groups','actor_rows','prefill_seconds','time')
def compact(r):return {k:r[k]for k in KEYS if k in r}
def good(r):
 return r.get('Adam_and_EMA_step_performed')is True and r.get('actual_Adam_updates')==r.get('EMA_updates')==r.get('cursor')and all(isinstance(r.get(k),(int,float))and math.isfinite(r[k])and r[k]>0 for k in('native_gradient_L2','head_gradient_L2'))and r.get('prefill_rows_equal_groups')is True

def six_plus_next(rows):
 if len(rows)<7:return False
 rows=list(rows)[-7:];return all(good(r)for r in rows)and all(b['actual_Adam_updates']==a['actual_Adam_updates']+1 and b['time']>a['time']for a,b in zip(rows,rows[1:]))
def run_cmd(args):
 try:
  p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=12);return dict(returncode=p.returncode,stdout=p.stdout.strip(),stderr=p.stderr[-1000:])
 except subprocess.TimeoutExpired:return dict(returncode=124,stdout='',stderr='Read-only scheduler query timed out; state unknown')
def log_tail(p):
 if not p.is_file():return None
 with p.open('rb')as f:f.seek(max(0,p.stat().st_size-5000));return f.read().decode(errors='replace')
def state_from(queue,accounting,job):
 if queue.get('returncode')==0:
  for line in queue['stdout'].splitlines():
   p=line.split('|')
   if p[0]==job:return p[1]
 if accounting.get('returncode')==0:
  for line in accounting['stdout'].splitlines():
   p=line.split('|')
   if p[0]==job:return p[1].split()[0]
 return 'UNKNOWN'
def validate_binding(r,e):
 b=r.get('binding',r);assert str(b.get('job_id',e['job_id']))==e['job_id'];assert b['init_seed']==e['init_seed']and b['train_plan_seed']==e['train_plan_seed'];assert b['fixture']==e['fixture']and b['physical_source_contract']==e['physical_source_contract'];assert b['training_questions']==169028 and b['layer']==15 and b['prefix']==80 and b['global_batch']==128 and b['objective']=='mse'and b['fixed_m']==5

def main_loop(base,deadline_hours=36):
 base=Path(base);assert base.is_dir();guard=(base/'observer.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);pub=read(base/'source-publication.json');assert proof(Path(__file__).resolve())==dict(pub['files'][pub['producer_relative']],path=str(Path(__file__).resolve()));jobs=read(Path(__file__).with_name('full169028_MSE_observer_jobs.json'));started=time.time();controller=dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],started=started,code=proof(Path(__file__).resolve()),source_publication=proof(base/'source-publication.json'),expected_jobs=jobs,training_files_read_only=True,Slurm_mutations=False,checkpoint_tensors_loaded=False,final4096_read=False);atomic(base/'controller.json',controller)
 histories={k:{str(r):collections.deque(maxlen=9)for r in(0,1)}for k in jobs};pairs={k:[]for k in jobs};confirmed={k:(base/('startup-proof-%s.json'%k)).is_file()for k in jobs};next_scheduler=0;next_snapshot=0;cached={};events=read(base/'phase-changes.json')if(base/'phase-changes.json').is_file()else[];last_phases={};terminal=set();errors={}
 # Bind exact immutable source/entry/readiness once; no checkpoint path is opened.
 for name,e in jobs.items():
  train=Path(e['base']);receipt=read(train/'root-submission.json');assert str(receipt['job_id'])==e['job_id']and receipt['source_commit']==e['source_commit'];checked(e['entry']);checked(e['ready']);sp=read(checked(e['source_publication']));assert sp['commit']==e['source_commit'];module=Path(sp['code_root'])/'reproduction/operations/qwen7b_warmup/online_packed_full169028_MSE_P80.py';assert proof(module)==dict(sp['files']['reproduction/operations/qwen7b_warmup/online_packed_full169028_MSE_P80.py'],path=str(module))
 while time.time()<started+deadline_hours*3600:
  now=time.time()
  if now>=next_scheduler:
   for name,e in jobs.items():
    queue=run_cmd(['squeue','-h','-j',e['job_id'],'-o','%i|%T|%M|%R']);accounting=run_cmd(['sacct','-n','-X','-j',e['job_id'],'--format=JobIDRaw,State,ExitCode,Elapsed,NodeList','-P']);cached[name]=dict(squeue=queue,sacct=accounting,state=state_from(queue,accounting,e['job_id']))
   next_scheduler=time.time()+30
  cells={};fast=False
  for name,e in jobs.items():
   train=Path(e['base']);cell=dict(job_id=e['job_id'],base=e['base'],source_commit=e['source_commit'],scheduler=cached[name],progress_by_rank={},observer_error=errors.get(name),startup_proven=confirmed[name]);failures={}
   try:
    if(train/'controller.json').is_file():
     c=read(train/'controller.json');validate_binding(c,e);assert str(c['job_id'])==e['job_id']and c['world_size']==2;cell['controller']={k:c[k]for k in('pid','job_id','init_seed','train_plan_seed','initial_head_sha256','world_size','time')}
    for rank in(0,1):
     p=train/('progress-rank-%d.json'%rank)
     if p.is_file():
      row=read(p);validate_binding(row,e);assert row['rank']==rank;r=compact(row);cell['progress_by_rank'][str(rank)]=r;h=histories[name][str(rank)]
      if not h or r['actual_Adam_updates']>h[-1]['actual_Adam_updates']:h.append(r)
     p=train/('failure-rank-%d.json'%rank)
     if p.is_file():failures[p.name]=read(p)
    if(train/'failure.json').is_file():failures['failure.json']=read(train/'failure.json')
    p=train/'complete.json'
    if p.is_file():cell['complete']=read(p)
    rs=cell['progress_by_rank']
    if len(rs)==2:
     pair=dict(time=time.time(),rank0=rs['0']['actual_Adam_updates'],rank1=rs['1']['actual_Adam_updates'],rank0_time=rs['0']['time'],rank1_time=rs['1']['time'])
     if not pairs[name]:pairs[name].append(pair)
     elif len(pairs[name])==1 and pair['rank0']>pairs[name][0]['rank0']and pair['rank1']>pairs[name][0]['rank1']:pairs[name].append(pair)
    if not confirmed[name]and len(pairs[name])==2 and all(six_plus_next(histories[name][str(r)])for r in(0,1)):
     p=dict(status='BOTH_RANKS_SIX_CONSECUTIVE_REAL_ADAM_AND_NEXT_STEP_PROVEN',time=time.time(),job_id=e['job_id'],source_commit=e['source_commit'],entry=e['entry'],ready=e['ready'],fixture=e['fixture'],physical_source_contract=e['physical_source_contract'],two_separate_actual_dual_rank_samples=pairs[name],rank_first_six={str(r):list(histories[name][str(r)])[-7:-1]for r in(0,1)},rank_next_step={str(r):list(histories[name][str(r)])[-1]for r in(0,1)},nonzero_finite_block_and_head_gradients=True,Adam_EMA_and_cursor_exact=True,training_files_modified=False,Slurm_mutations=False,final4096_read=False);atomic(base/('startup-proof-%s.json'%name),p);confirmed[name]=True;cell['startup_proven']=True
    state=cached[name]['state'];cell['failures']=failures
    if failures or state in('FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE'):
     if name not in errors:
      errors[name]=dict(status='ACTUAL_TRAINING_FAILURE_REQUIRES_ROOT_ATTENTION',time=time.time(),job_id=e['job_id'],state=state,failures=failures,scheduler=cached[name],last_progress=rs,log_tail=log_tail(train/('train-%s.log'%e['job_id'])),no_training_mutation=True,final4096_read=False);atomic(base/('failure-%s.json'%name),errors[name])
     terminal.add(name)
    elif state=='COMPLETED':
     if not cell.get('complete')or cell['complete'].get('actual_Adam_updates')!=13210:cell['completed_without_complete_acceptance']=True
     terminal.add(name)
    if state=='RUNNING'and not confirmed[name]and name not in terminal:fast=True
    cell['bounded_rank_history']={r:list(h)for r,h in histories[name].items()};cell['observer_error']=errors.get(name)
    phase=(state,rs.get('0',{}).get('epoch'),rs.get('0',{}).get('stage'),confirmed[name],bool(failures),bool(cell.get('complete')))
    if last_phases.get(name)!=phase:
     events.append(dict(time=time.time(),replica=name,job_id=e['job_id'],phase=list(phase),rank_cursors={r:v['cursor']for r,v in rs.items()}));events=events[-96:];atomic(base/'phase-changes.json',events);last_phases[name]=phase
   except BaseException:
    cell['read_or_binding_error']=dict(time=time.time(),traceback=traceback.format_exc(),training_state_not_inferred=True);atomic(base/('metadata-read-error-%s.json'%name),cell['read_or_binding_error'])
   cells[name]=cell
  if time.time()>=next_snapshot:
   atomic(base/'latest.json',dict(time=time.time(),replicas=cells,observer_pid=os.getpid(),training_files_read_only=True,Slurm_mutations=False,checkpoint_tensors_loaded=False,final4096_read=False));next_snapshot=time.time()+(1 if fast else 25)
  if len(terminal)==len(jobs):atomic(base/'complete.json',dict(status='EXACT_TWO_TRAINING_JOBS_METADATA_OBSERVATION_TERMINAL',time=time.time(),terminal_jobs=[e['job_id']for e in jobs.values()],startup_proven=confirmed,training_failures_present=bool(errors),training_goal_completion_not_claimed=True,final4096_read=False));return
  time.sleep(.20 if fast else 5)
 atomic(base/'deadline.json',dict(status='OBSERVER_BOUNDED_DEADLINE_REACHED_NO_TRAINING_ACTION',time=time.time(),final4096_read=False))
def main():
 p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--deadline-hours',type=float,default=36);a=p.parse_args()
 try:main_loop(a.base,a.deadline_hours)
 except BaseException:
  if a.base.is_dir():atomic(a.base/'observer-failure.json',dict(time=time.time(),traceback=traceback.format_exc(),training_state_not_inferred=True,Slurm_mutations=False,final4096_read=False))
  raise
if __name__=='__main__':main()
