# Historical exact operator executed on 2026-10-22; no credentials.
import subprocess,json,shlex,time,os
from pathlib import Path
b=Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fastpath-probe-20261022');prior=b/'diagnostics/candidate-cpu-cause-r1';root=b/'diagnostics/candidate-cpu-cause-step-r2';root.mkdir(parents=True,exist_ok=True)
if (root/'launch-receipt.json').exists():print((root/'launch-receipt.json').read_text());raise SystemExit()
commands=json.loads((prior/'submit-claim.json').read_text())['commands']
commands=[[a.replace(str(prior),str(root)) for a in c] for c in commands]
wrapper=' && '.join(shlex.join(c) for c in commands)
argv=['srun','--jobid=745198','--overlap','--exact','--ntasks=1','--cpus-per-task=2','--mem=32G','--gres=none','--cpu-bind=none','--chdir=/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fastpath-probe-20261022/source/665704a6c3a02a4b92e6a441b9543363097edf5a','/bin/sh','-c',wrapper]
with (root/'step.log').open('x') as stream:p=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
receipt={'epoch':time.time(),'pid':p.pid,'start_ticks':Path('/proc/%d/stat'%p.pid).read_text().split()[21],'argv':argv,'commands':commands,'allocation':745198,'cpu_only_step':True,'gpu_resources_requested':0,'production_runtime_modified':False,'prior_rejection_preserved':str(prior/'submit-receipt.json')}
(root/'launch-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:receipt[k] for k in ('epoch','pid','start_ticks','allocation','cpu_only_step','gpu_resources_requested','production_runtime_modified')}))
