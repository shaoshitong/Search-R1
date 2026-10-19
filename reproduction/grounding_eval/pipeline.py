"""One explicitly bound checkpoint: normal NQ -> frozen CF -> independent verify.

Start only after the first checkpoint's real protocol pilot has been reviewed.
No discovery, training, GPU selection or external submission happens here.
"""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

from .core import atomic, digest, read, utc

def main():
    ap=argparse.ArgumentParser(__doc__);ap.add_argument('--job',required=True);args=ap.parse_args()
    job=read(args.job);root=Path(job['pipeline_output']);root.mkdir(parents=True,exist_ok=True)
    lock=open(root/'pipeline.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    job_hash=digest(job)
    if (root/'binding.json').exists():
        if read(root/'binding.json')!=job:raise ValueError('Changed pipeline binding')
    else:atomic(root/'binding.json',job)
    identity=dict(pid=os.getpid(),command=sys.argv,job_hash=job_hash,started_at=utc())
    atomic(root/'worker.json',identity,replace=True)
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONNOUSERSITE='1')
    env['PYTHONPATH']=job['runtime']
    def step(name,command):
        phase=dict(phase=name,command=command,started_at=utc(),job_hash=job_hash)
        atomic(root/'progress.json',phase,replace=True)
        with open(root/(name+'.log'),'ab',buffering=0) as log:
            p=subprocess.Popen(command,cwd=job['runtime'],env=env,stdout=log,stderr=subprocess.STDOUT)
            phase['child_pid']=p.pid;atomic(root/'progress.json',phase,replace=True)
            code=p.wait()
        atomic(root/(name+'-exit.json'),dict(phase,exit_code=code,completed_at=utc()),replace=True)
        if code:raise RuntimeError(name+' failed with exit '+str(code))
    python=job['python'];module=[python,'-m','reproduction.grounding_eval']
    try:
        step('normal',job['normal_command'])
        if job.get('export_a800'):
            step('export-normal',[python,'-m','reproduction.grounding_eval.export_normal',
                '--normal-root',job['normal_root'],'--count','3610','--output',job['source']])
        cf=Path(job['cf_output']);manifest=cf/'manifest.json'
        if not manifest.exists():
            command=module+['prepare','--source',job['source'],'--checkpoint',job['checkpoint'],
                '--output',str(cf),'--method',job['method'],'--run-id',job['run_id'],
                '--normal-config',job['normal_config']]
            if job.get('seed') is not None:command+=['--seed',str(job['seed'])]
            if job.get('pilot_cache'):command+=['--reuse-results',job['pilot_cache']]
            step('prepare-cf',command)
        elif read(manifest)['method']!=job['method'] or read(manifest)['run_id']!=job['run_id']:
            raise ValueError('Existing CF manifest differs from pipeline method/run')
        resource=['--max-batch',str(job.get('cf_max_batch',256)),
                  '--max-memory-utilization',str(job.get('cf_max_memory_utilization',.9))]
        if job.get('gpu_uuid'):resource+=['--gpu-uuid',job['gpu_uuid']]
        step('counterfactual',module+['run','--manifest',str(manifest)]+resource)
        step('verify',module+['verify','--manifest',str(manifest)])
        summary=read(cf/'verified-summary.json')
        if summary['status']!='complete' or summary['counts']['normal_questions']!=3610:
            raise RuntimeError('Evaluation remains provisional/unavailable; inspect coverage before acceptance')
        atomic(root/'complete.json',dict(identity,completed_at=utc(),exit_code=0,
            verified_summary=str(cf/'verified-summary.json'),summary=summary),replace=True)
        atomic(root/'progress.json',dict(phase='COMPLETE',completed_at=utc(),job_hash=job_hash),replace=True)
    except Exception as error:
        atomic(root/'failed.json',dict(identity,failed_at=utc(),error=type(error).__name__+':'+str(error)),replace=True)
        raise

if __name__=='__main__':main()
