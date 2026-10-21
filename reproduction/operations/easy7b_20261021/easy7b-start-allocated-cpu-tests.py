import json,time,subprocess,os
from pathlib import Path
b=Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-20261021');c=b/'optimizations/selected-head-63c7ba2a-placeholder'
roots=list((b/'optimizations').glob('selected-head-*'));assert len(roots)==1;c=roots[0]
claim=c/'allocated-cpu-test-claim.json';assert not claim.exists(),'Observe existing test step, do not duplicate'
job=(b/'job.id').read_text().strip();assert job=='745022'
x=subprocess.run(['squeue','-j',job,'-h','-o','%i|%u|%T'],capture_output=True,text=True,timeout=8);assert x.returncode==0 and x.stdout.strip()==job+'|jxu957|RUNNING'
cmd=['srun','--jobid='+job,'--overlap','--exact','--ntasks=1','--cpus-per-task=2','--gres=gpu:0','env','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=2','OPENBLAS_NUM_THREADS=2','PYTHONPATH='+str(c),'/data/user/jxu957/experiments/topd-20261008/runtime/env/bin/python','-B','-m','unittest','search_g1.test_selected_qwen_head','-v']
wrapper='''import subprocess,json,time,os,socket
from pathlib import Path
c=Path(ROOT);s=time.time()
with(c/'allocated-cpu-test.stdout').open('x')as o,(c/'allocated-cpu-test.stderr').open('x')as e:r=subprocess.run(COMMAND,cwd=c,stdout=o,stderr=e,timeout=300)
(c/'allocated-cpu-test-result.json').write_text(json.dumps(dict(epoch=time.time(),seconds=time.time()-s,returncode=r.returncode,command=COMMAND,GPU_allocated=0)))
'''.replace('ROOT',repr(str(c))).replace('COMMAND',repr(cmd))
with(c/'allocated-cpu-test-wrapper.out').open('x')as o,(c/'allocated-cpu-test-wrapper.err').open('x')as e:
 p=subprocess.Popen(['/data/user/jxu957/experiments/topd-20261008/runtime/env/bin/python','-B','-c',wrapper],stdout=o,stderr=e,stdin=subprocess.DEVNULL,start_new_session=True,cwd=c)
v=dict(epoch=time.time(),pid=p.pid,birth=Path('/proc/%d/stat'%p.pid).read_text().rsplit(')',1)[1].split()[19],uid=os.getuid(),command=cmd,root=str(c),previous_test='login CPU child timed out40s; no numeric acceptance')
with claim.open('x')as f:json.dump(v,f,indent=2);f.flush();os.fsync(f.fileno())
print(json.dumps(v))
