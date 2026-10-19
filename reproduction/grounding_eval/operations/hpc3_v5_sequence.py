"""Task-only HPC3 CPU prepare or one-GPU sequential CF of frozen v5 manifests."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


def atomic(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    with tmp.open('x') as f:
        json.dump(value,f,ensure_ascii=False,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(str(tmp),str(path))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('action',choices=('prepare','run'))
    p.add_argument('--job',required=True);p.add_argument('--max-batch',type=int,default=256)
    args=p.parse_args();job=json.loads(Path(args.job).read_text());root=Path(job['sequence_output']);root.mkdir(parents=True,exist_ok=True)
    lock=(root/(args.action+'.lock')).open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    atomic(root/(args.action+'-worker.json'),dict(pid=os.getpid(),host=socket.gethostname(),at=time.time(),
        command=sys.argv,slurm_job_id=os.environ.get('SLURM_JOB_ID'),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')))
    if args.action=='run':
        if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('GPU inference requires an actual Slurm allocation')
        if not Path(job['root_gate']).is_file():raise RuntimeError('Root G1 v5 acceptance gate absent; no GPU execution')
        gate=json.loads(Path(job['root_gate']).read_text())
        if gate.get('accepted') is not True or gate.get('protocol_version')!='strict-v5-conservative-typed':raise RuntimeError('Wrong gate/protocol')
    for entry in job['methods']:
        base=[job['python'],'-B','-m','reproduction.grounding_eval.strict_v5'];out=Path(entry['cf_output'])
        commands=[]
        if args.action=='prepare':
            if (out/'manifest.json').exists():
                raise RuntimeError('Existing manifest: inspect/resume explicitly, never overwrite prepared outputs')
            commands.append(base+['prepare','--source',entry['source'],'--normal-config',entry['normal_config'],
                '--checkpoint',entry['checkpoint'],'--output',str(out),'--method',entry['method'],
                '--run-id',entry['run_id'],'--expected-count','3610'])
            if entry.get('seed') is not None:commands[-1]+= ['--seed',str(entry['seed'])]
        else:
            commands.extend([base+['run','--manifest',str(out/'manifest.json'),'--max-batch',str(args.max_batch),
                '--max-memory-utilization',str(job['cf_max_memory_utilization'])],
                base+['verify','--manifest',str(out/'manifest.json')]])
        for n,command in enumerate(commands):
            log=root/(entry['key']+'-'+args.action+'-'+str(n)+'.log')
            started=time.time()
            with log.open('ab',buffering=0) as stream:
                proc=subprocess.Popen(command,cwd=job['runtime'],stdout=stream,stderr=subprocess.STDOUT)
                atomic(root/'active-child.json',dict(pid=proc.pid,parent_pid=os.getpid(),method=entry['method'],command=command,at=started))
                status=proc.wait()
            atomic(root/(entry['key']+'-'+args.action+'-'+str(n)+'-exit.json'),dict(exit_code=status,started=started,ended=time.time(),command=command))
            if status:raise RuntimeError('Own evaluation subprocess failed; preserve outputs and inspect '+str(log))
        summary=json.loads((out/('verified-summary.json' if args.action=='run' else 'summary.json')).read_text())
        atomic(root/(entry['key']+'-'+args.action+'-summary.json'),summary)
    atomic(root/(args.action+'-complete.json'),dict(exit_code=0,at=time.time(),methods=[x['method'] for x in job['methods']]))


if __name__=='__main__':main()
