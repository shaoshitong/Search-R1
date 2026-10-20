"""Own empty GPUs 1/2/3: two TP1 generators, one shared local E5/Flat GPU."""
import argparse,json,os,signal,subprocess,sys,time,urllib.request
from pathlib import Path
from reproduction.operations.qwen7b_warmup.download_model import atomic
from reproduction.h1008_qprobe.pipeline import UUIDLocks

def main():
 p=argparse.ArgumentParser();p.add_argument('--base',required=True);a=p.parse_args();b=Path(a.base);c=json.loads((b/'config.json').read_text());logs=b/'logs'/os.environ.get('Q7B_ATTEMPT','r3');logs.mkdir(parents=True,exist_ok=True)
 rows=[x.strip().split(', ')for x in subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],text=True).splitlines()];gpu={int(x[0]):x for x in rows};assert all(int(gpu[i][2])<64 for i in [1,2,3]),'Assigned cards are no longer empty'
 lease=UUIDLocks([gpu[i][1]for i in [1,2,3]]);children=[];handles=[]
 def stop(*_):
  for proc in children:
   if proc.poll()is None:proc.terminate()
  for proc in children:
   try:proc.wait(timeout=30)
   except subprocess.TimeoutExpired:proc.kill();proc.wait()
  lease.close()
 signal.signal(signal.SIGTERM,lambda *_:(stop(),sys.exit(143)))
 atomic(b/'controller.json',dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],run_id=c['run_id'],cards={i:gpu[i][1]for i in [1,2,3]},seed=c['seed'],epoch=time.time(),actor_updates=0,probe_updates=0))
 def spawn(name,args,visible):
  f=(logs/(name+'.log')).open('ab');handles.append(f);env=dict(os.environ,CUDA_VISIBLE_DEVICES=visible);proc=subprocess.Popen([sys.executable,'-u','-m',*args],env=env,stdout=f,stderr=f);children.append(proc);return proc
 try:
  retr=spawn('retriever',['reproduction.operations.qwen7b_warmup.retriever','--index',c['retrieval_index'],'--corpus',c['retrieval_corpus'],'--encoder',str(b.parent/'search-g1-prefix5-h10011-runtime-assets-20261018/common8/assets/e5-base-v2'),'--data-cache',c['retrieval_data_cache'],'--cache',str(b/'retrieval'),'--run-id',c['run_id'],'--port','28831','--batch-queries','512'],'3');start=time.monotonic()
  while True:
   if retr.poll()is not None:raise RuntimeError('Owned retriever exited: see logs/retriever.log')
   try:
    with urllib.request.urlopen('http://127.0.0.1:28831/health',timeout=2)as f:v=json.load(f)
    if v.get('ready'):break
   except (OSError,ValueError):pass
   if time.monotonic()-start>1800:raise TimeoutError('Retriever initialization exceeded 30min')
   atomic(b/'progress.json',dict(stage='STARTING_SHARED_RETRIEVER',epoch=time.time()));time.sleep(2)
  workers=[spawn('sampling-%02d'%i,['reproduction.operations.qwen7b_resample.worker','--base',str(b),'--shard',str(i)],'%d,3'%(i+1))for i in range(2)]
  atomic(b/'children.json',dict(retriever_pid=retr.pid,worker_pids=[p.pid for p in workers],attempt=os.environ.get('Q7B_ATTEMPT'),epoch=time.time()))
  while any(p.poll()is None for p in workers):
   if retr.poll()is not None:raise RuntimeError('Owned retrieval service exited during sampling')
   for proc in workers:
    if proc.poll()not in(None,0):raise RuntimeError('Own sampling child exited %d'%proc.returncode)
   progress=[]
   for i in range(2):
    f=b/'sampling'/('shard-%02d'%i)/'progress.json'
    if f.exists():progress.append(json.loads(f.read_text()))
   atomic(b/'progress.json',dict(stage='SAMPLING_NEW_G5',questions=sum(x['questions']for x in progress),trajectories=sum(x['trajectories']for x in progress),expected_questions=169028,expected_trajectories=845140,shards=progress,epoch=time.time()));time.sleep(15)
  exits=[p.wait()for p in workers];atomic(b/'worker-exits.json',dict(returncodes=exits,epoch=time.time()))
  if any(exits):raise RuntimeError('Sampling worker terminal returncodes: '+repr(exits))
  assert all((b/'sampling'/('shard-%02d'%i)/'complete.json').is_file()for i in range(2)),'Worker exited without a sealed shard; inspect original sampling logs'
  result=[json.loads((b/'sampling'/('shard-%02d'%i)/'complete.json').read_text())for i in range(2)];assert sum(x['questions']for x in result)==169028 and sum(x['trajectories']for x in result)==845140
  atomic(b/'complete.json',dict(status='ALL_169028_QUESTIONS_NEW_SEED_G5_COMPLETE',run_id=c['run_id'],seed=c['seed'],questions=169028,trajectories=845140,hidden_cached=False,actor_updates=0,probe_updates=0,shards=result,epoch=time.time()))
 except BaseException as error:atomic(b/'failure.json',dict(error_type=type(error).__name__,error=str(error)[:300],epoch=time.time()));raise
 finally:
  stop()
  for f in handles:f.close()
if __name__=='__main__':main()
