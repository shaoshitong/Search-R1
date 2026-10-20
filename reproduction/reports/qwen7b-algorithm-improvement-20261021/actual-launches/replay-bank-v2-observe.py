"""Read-only R2 capture/Slurm/locks/byte observations; no submissions or fixes."""
import pathlib,json,hashlib,subprocess,fcntl,time
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/replay-matched9088-bank-v2-20261021')
expected_preparation='a7fa015dd655ef40d371172375ae30e4280503375485bd61ed6ab49baf5c1fd9'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def command(args):
 r=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=20)
 return dict(returncode=r.returncode,stdout=r.stdout.strip(),stderr=r.stderr[-1000:])
def lockstate(p):
 if not p.exists():return 'ABSENT'
 with p.open('r+') as f:
  try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return 'HELD'
  fcntl.flock(f,fcntl.LOCK_UN);return 'FREE'
assert sha(b/'prepared/preparation.json')==expected_preparation
records=[];ids=[]
for i in range(4):
 root=b/'prepared'/('shard-%02d'%i);cap=root/'capture';job=(root/'job.id').read_text().strip() if (root/'job.id').exists() else None
 row=dict(shard=i,job_id=job,run_lock=lockstate(root/'replay-run.lock'),capture_lock=lockstate(cap/'capture.lock'),capture_exists=cap.exists())
 if job:
  assert job.isdigit();ids.append(job);s=command(['scontrol','show','job',job,'-o']);row['scontrol']=s
  if s['returncode']==0:
   text=s['stdout'];assert 'JobName=q7b-replay-bank-'+str(i)+' ' in text and 'Command='+str(root/'replay.sbatch') in text
  row['accounting']=command(['sacct','-n','-P','-j',job,'-o','JobID,JobName,State,ExitCode,Elapsed,Start,End'])
 for name in ['progress.json','failure.json','complete.json']:
  path=cap/name
  if path.exists():
   v=read(path)
   row[name]=v if name!='complete.json' else {k:v.get(k) for k in ('status','questions','trajectories','capture_layers','native_bitexact_equivalence_claim','elapsed_seconds')}
 if cap.exists():
  files=[p for p in cap.rglob('*') if p.is_file() and p.name in ('question.bin','delta.bin','policy.bin','prototype.bin','index.jsonl')]
  row['actual_capture_file_bytes']=sum(p.stat().st_size for p in files)
  row['actual_capture_payload_files']=len(files)
  complete=sorted(cap.glob('chunk-*/complete.json'))
  row['completed_chunks']=len(complete)
  row['completed_chunk_trajectories']=sum(len(read(p)['trajectory_ids']) for p in complete)
 if job:
  logs=sorted((root/'logs').glob('replay-'+job+'.log'))
  if logs:
   p=logs[-1];size=p.stat().st_size
   with p.open('rb') as f:f.seek(max(0,size-5000));raw=f.read(5000)
   row['log']=dict(path=str(p),bytes=size,tail=raw.decode(errors='replace'))
 records.append(row)
print(json.dumps(dict(status='ACTUAL_READONLY_REPLAY_BANK_SHARD_OBSERVATION',time=time.time(),base=str(b),source_commit='65f08f72b4e61cf72e3c43a202bda0508830a160',rows=records,queue=command(['squeue','-h','-u','mou176,jxu957','-o','%i|%j|%T|%P|%R|%b']),estimates=command(['squeue','--start','-h','-j',','.join(ids),'-o','%i|%j|%T|%S|%R']) if ids else None)))
