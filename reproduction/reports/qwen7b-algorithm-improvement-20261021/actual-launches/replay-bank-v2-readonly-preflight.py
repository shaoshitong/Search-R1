import pathlib,json,hashlib,subprocess,fcntl,time
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/replay-matched9088-bank-v2-20261021')
commit='65f08f72b4e61cf72e3c43a202bda0508830a160'
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
 return h.hexdigest()
def command(args):
 r=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=20);assert r.returncode==0,r.stderr;return r.stdout.strip()
def lockstate(p):
 if not p.exists():return 'ABSENT'
 with p.open('r+') as f:
  try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return 'HELD'
  fcntl.flock(f,fcntl.LOCK_UN);return 'FREE'
prep=b/'prepared/preparation.json';assert sha(prep)=='a7fa015dd655ef40d371172375ae30e4280503375485bd61ed6ab49baf5c1fd9'
cpu=read(b/'cpu-result.json');assert cpu['status']=='ACTUAL_REAL_9088Q_FOUR_SHARD_CPU_FIXTURES_PREPARED_NOT_CAPTURED'
pub=read(b/'source-publication.json');assert pub['commit']==commit
code=pathlib.Path(pub['code_root']);assert code==b/('code-'+commit[:12])
for name,v in pub['files'].items():assert (code/name).stat().st_size==v['bytes'] and sha(code/name)==v['sha256']
prepared=read(prep);assert prepared['per_shard_questions']==[2272]*4 and prepared['per_shard_trajectories']==[11360]*4
rows=[]
for i in range(4):
 root=b/'prepared'/('shard-%02d'%i)
 row=dict(shard=i,base=str(root),run_lock=lockstate(root/'replay-run.lock'),capture_lock=lockstate(root/'capture/capture.lock'))
 for name in ['job.id','submission-intent.json','capture/complete.json','capture/failure.json','capture/progress.json']:
  p=root/name
  row[name]=(p.read_text().strip() if name=='job.id' else read(p)) if p.exists() else None
 for ref in [prepared['selections'][i],prepared['scripts'][i]]:
  p=pathlib.Path(ref['path']);assert p.stat().st_size==ref['bytes'] and sha(p)==ref['sha256']
 rows.append(row)
print(json.dumps(dict(status='R2_ACTUAL_READONLY_SOURCE_FIXTURE_WRITER_PREFLIGHT',time=time.time(),preparation_sha256=sha(prep),source_commit=commit,account_active=command(['squeue','-h','-u','mou176,jxu957','-o','%i|%j|%T|%P|%R|%b']),rows=rows)))
