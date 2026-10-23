"""Run the two already-published selected-epoch acceptors sequentially; no tensors here."""
import fcntl,hashlib,json,os,pathlib,subprocess,time
A=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021')
OUT=A/'full169028-L15P80-MSE-selected-CPU-epoch09-20261023-v1'
ACCEPT=A/'full169028-L15P80-MSE-terminal-CPU-acceptance-20261023-v1'
CODE=ACCEPT/'code-693a81dfbbae'
SELECT=A/'full169028-P80-MSE-actual-family-selection-20261023-v1/actual-family-selection-and-paired.json'
def read(p):return json.loads(p.read_bytes())
def proof(p):
 raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def atomic(p,v):
 t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,sort_keys=True)+'\n');t.replace(p)
def ticks(pid):return pathlib.Path('/proc/%d/stat'%pid).read_text().split(') ',1)[1].split()[19]
OUT.mkdir(parents=True,exist_ok=True)
guard=(OUT/'controller.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert proof(SELECT)['sha256']=='0db5e48b4ee7f06ed44c111f67fac2f7615ca08b087e4b799ecdcd3f6dce878f'
selection=read(SELECT);assert selection['actual_selected']['epoch']==9 and selection['actual_selected']['state']=='raw'
assert proof(CODE/'accept_full169028_L15P80_MSE_checkpoint.py')['sha256']=='40e308e6a3c65bc0132c7b249939059775f4513d867576e8df74fa7aabc61448'
accounting=subprocess.check_output(['sacct','-n','-X','-P','-j','750189,750190','--format=JobIDRaw,State,ExitCode'],universal_newlines=True,timeout=20)
for job in ('750189','750190'):assert job+'|COMPLETED|0:0' in accounting
atomic(OUT/'controller.json',dict(pid=os.getpid(),start_ticks=ticks(os.getpid()),selected_epoch=9,state='raw',selection=proof(SELECT),time=time.time()))
results=[]
for seed,plan,job in ((9137,2026102227,'750189'),(19137,2026102302,'750190')):
 base=A/('full169028-L15P80-MSE-ownG5-physical-seed%d-plan%d-20261023-v3'%(seed,plan))
 receipt=ACCEPT/base.name/'epoch-09/accepted.json';entry=CODE/('seed%d.sh'%seed)
 if not receipt.is_file():
  mutex=(A/'online-mixed-P80-handoff-CPU-acceptance-20261023-v1/acceptance.lock').open('a+')
  fcntl.flock(mutex,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(mutex,fcntl.LOCK_UN);mutex.close()
  with(OUT/('seed%d-epoch09.log'%seed)).open('ab')as log:
   child=subprocess.Popen(['bash',str(entry),'--epoch','9'],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,close_fds=True)
   atomic(OUT/'progress.json',dict(stage='VALIDATING_EXISTING_SELECTED_CHECKPOINT',seed=seed,job_id=job,child_pid=child.pid,child_start_ticks=ticks(child.pid),entry=proof(entry),time=time.time()))
   code=child.wait()
  assert code==0,'Existing immutable selected-epoch validator failed; inspect its log before any retry'
 accepted=read(receipt)
 assert accepted['job_id']==job and accepted['epoch']==9 and accepted['actual_Adam_updates']==accepted['EMA_updates']==11889
 assert accepted['epoch_complete'] and accepted['next_epoch']==10 and accepted['within_next_epoch_question_position']==0
 assert accepted['CPU_only'] and not accepted['GPU_forward_or_next_update_restore_tested'] and not accepted['final4096_read']
 assert accepted['checkpoint']['path']==str(base/'checkpoint-epoch-09.pt')
 results.append(dict(seed=seed,job_id=job,accepted=proof(receipt),checkpoint=accepted['checkpoint']))
 atomic(OUT/'progress.json',dict(stage='SELECTED_EPOCH_ACCEPTED',results=results,time=time.time()))
result=dict(status='BOTH_ACTUAL_SELECTED_EPOCH9_CPU_ACCEPTED',selected_epoch=9,state='raw',selection=proof(SELECT),results=results,Slurm_mutated=False,training_changed=False,final4096_read=False,time=time.time())
atomic(OUT/'complete.json',result);print(json.dumps(result))
