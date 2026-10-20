"""Start ONCE the already published65f08 CPU full-SHA bank merge, after4successes."""
import pathlib,json,hashlib,subprocess,fcntl,time,os,shlex
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/replay-matched9088-bank-v2-20261021')
commit='65f08f72b4e61cf72e3c43a202bda0508830a160';code=b/('code-'+commit[:12]);prepared=b/'prepared'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def atomic(p,v):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
def live(pid,ticks):
 try:return pathlib.Path('/proc/%d/stat'%pid).read_text().rsplit(')',1)[1].split()[19]==str(ticks)
 except OSError:return False
lock=(b/'full-sha-merge-controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert sha(prepared/'preparation.json')=='a7fa015dd655ef40d371172375ae30e4280503375485bd61ed6ab49baf5c1fd9'
if (prepared/'bank-complete.json').exists():
 done=read(prepared/'bank-complete.json');assert done['status']=='ALL_FOUR_MATCHED9088Q_MULTILAYER_HF_REPLAY_BANKS_SHA_VERIFIED'
 print(json.dumps(dict(status='EXISTING_REAL_FULL_SHA_MERGE_COMPLETE_NOT_RESTARTED',bank_complete=ref(prepared/'bank-complete.json'))));raise SystemExit(0)
if (b/'full-sha-merge-intent.json').exists():
 launch=read(b/'full-sha-merge-launch.json') if (b/'full-sha-merge-launch.json').exists() else None
 print(json.dumps(dict(status='EXISTING_FULL_SHA_MERGE_INTENT_INSPECT_DO_NOT_RESTART',launch=launch,identity_live=live(launch['pid'],launch['start_ticks']) if launch else False)));raise SystemExit(0)
pub=read(b/'source-publication.json');assert pub['commit']==commit and pub['code_root']==str(code)
for name,v in pub['files'].items():assert (code/name).stat().st_size==v['bytes'] and sha(code/name)==v['sha256']
prep=read(prepared/'preparation.json');success=[]
for i,job in enumerate(['743087','743088','743089','743090']):
 root=prepared/('shard-%02d'%i);assert (root/'job.id').read_text().strip()==job
 done=root/'capture/complete.json';d=read(done);sel=prep['selections'][i]
 assert d['status']=='MATCHED_REAL_G5_MULTILAYER_HF_CAUSAL_REPLAY_SHARD_COMPLETE' and d['source_selection']==sel
 assert d['questions']==2272 and d['trajectories']==11360 and d['capture_layers']==[20,23,26]
 assert d['actor_updates']==0 and not (root/'capture/failure.json').exists()
 r=subprocess.run(['sacct','-n','-P','-j',job,'-o','JobID,State,ExitCode'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=20)
 assert r.returncode==0,r.stderr
 rows=[line.split('|') for line in r.stdout.splitlines() if line.startswith(job+'|')]
 assert len(rows)==1 and rows[0][1:3]==['COMPLETED','0:0'],r.stdout
 success.append(dict(shard=i,job=job,completion=ref(done),accounting=rows[0]))
# Small controller-only shell wrapper; the SHA implementation is the existing
# immutable replay_bank_prepare module, not copied or monkeypatched here.
exit_py="import pathlib,json,sys,time; b=pathlib.Path(sys.argv[1]); rc=int(sys.argv[2]); p=b/'full-sha-merge-exit.json'; t=p.with_suffix('.tmp'); t.write_text(json.dumps(dict(returncode=rc,status='FULL_SHA_MERGE_COMMAND_EXITED',time=time.time()))+'\\n'); t.replace(p)"
command='''set +e
exec 9>@@MERGE_LOCK@@
flock -n 9 || exit 73
bash @@MERGE_RUNTIME@@ -u -m reproduction.operations.qwen7b_warmup.replay_bank_prepare --base @@MERGE_PREPARED@@ --merge >@@MERGE_LOG@@ 2>&1
merge_rc=$?
python3 -c @@MERGE_EXIT_CODE@@ @@MERGE_BASE@@ "$merge_rc"
exit "$merge_rc"
'''
for token,value in [('@@MERGE_LOCK@@',b/'full-sha-merge.lock'),('@@MERGE_RUNTIME@@',code/'reproduction/operations/qwen7b_warmup/runtime.sh'),('@@MERGE_PREPARED@@',prepared),('@@MERGE_LOG@@',b/'full-sha-merge.log'),('@@MERGE_EXIT_CODE@@',exit_py),('@@MERGE_BASE@@',b)]:
 command=command.replace(token,shlex.quote(str(value)))
intent=dict(source_commit=commit,source_preparation=ref(prepared/'preparation.json'),source_publication=ref(b/'source-publication.json'),
 command_sha256=hashlib.sha256(command.encode()).hexdigest(),completed_capture_jobs=success,time=time.time(),GPU_requested=False,student_updates=0)
atomic(b/'full-sha-merge-intent.json',intent)
env=dict(os.environ,QWEN7B_CODE_ROOT=str(code),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
with (b/'full-sha-merge-controller.log').open('ab') as log:
 p=subprocess.Popen(['bash','-c',command],cwd=str(code),env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 ticks=pathlib.Path('/proc/%d/stat'%p.pid).read_text().rsplit(')',1)[1].split()[19]
launch=dict(pid=p.pid,start_ticks=ticks,time=time.time(),source_commit=commit,source_preparation=intent['source_preparation'],command_sha256=intent['command_sha256'],GPU_requested=False)
atomic(b/'full-sha-merge-launch.json',launch)
print(json.dumps(dict(status='OWNED_EXISTING65F08_CPU_FULL_SHA_MERGE_LAUNCHED_VERIFY_PROGRESS',launch=launch)))
