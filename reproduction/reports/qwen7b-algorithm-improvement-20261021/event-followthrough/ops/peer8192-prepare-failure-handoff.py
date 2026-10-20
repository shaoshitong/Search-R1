import pathlib,json,os,fcntl,time,subprocess
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/peer-newseed8192-g10-train-v1');r=lambda p:json.loads(p.read_text());locks=[]
for name in ['deployment.lock','peer8192.lock','E4-baseline.lock']:
 h=(b/name).open('a+');fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(h)
assert not any((b/n).exists() for n in ['job.id','E4-baseline.job.id','config.json','preparation.json','E4-baseline-preparation.json','latest.json','inputs/E4-only-config.json'])
pub=r(b/'source-publication.json');assert pub['commit']=='e580c26824dbf3a13705db0a7a92ab3a0df31e66'
c=r(b/'E4-cpu-prepare-controller.json');assert c['pid']==882605
try:cmd=pathlib.Path('/proc/%d/cmdline'%c['pid']).read_bytes().decode(errors='replace')
except OSError:cmd=''
assert not ('peer_newseed_fit_operation' in cmd and str(b) in cmd),'Actual CPU writer still active'
for name in ['q7b-peer8192-g10-hf-fit','q7b-peer-E4-same-HF-val']:
 p=subprocess.run(['squeue','-h','-u','mou176','-n',name,'-o','%i|%j|%T'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=20);assert p.returncode==0 and not p.stdout.strip(),p.stdout
log=(b/'E4-cpu-prepare.log').read_text();assert "KeyError: 'bytes'" in log
a=b/'failed-cpu-preparation-e580c26824db';a.mkdir(exist_ok=False);moved=[]
for name in ['source-publication.json','upload-controller.json','E4-cpu-prepare-controller.json','E4-cpu-prepare.log','actual-cpu-tests.log']:
 p=b/name
 if p.exists():os.rename(str(p),str(a/name));moved.append(name)
v={'status':'FAILED_OWN_CPU_PREPARATION_PRESERVED_FOR_PUBLISHED_SCHEMA_FIX','old_commit':pub['commit'],'old_pid':c['pid'],'moved_small_metadata':moved,'old_immutable_code_preserved':pub['code_root'],'model_or_checkpoint_modified':False,'GPU_jobs_submitted':False,'reason':'Legacy SHA-only metadata descriptor omitted bytes; published fix uses actual stat bytes plus original SHA','time':time.time()}
(a/'handoff.json').write_text(json.dumps(v,sort_keys=True));print(json.dumps(v))
