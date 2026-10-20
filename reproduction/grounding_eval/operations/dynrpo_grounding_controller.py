"""Sequential NQ native-export, frozen-v4 diagnostic, conservative-v5, CPU acceptance."""
import argparse,fcntl,json,os,subprocess,sys,time,socket
from pathlib import Path
from reproduction.grounding_eval.core import atomic,read,sha
ROOT=Path('/data/user/jxu957/experiments/dynrpo-nq-grounding-20261020')
BASE='/data/user/jxu957/experiments/dynrpo-prefix5-thread5-20261017-r4'
DATA='/data/user/jxu957/experiments/topd-20261008/runtime/assets/common-instruct-0b002b/final.parquet'
RUN='cc0de51f2d68495c9664aa888074839c'
def main():
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES')
    root=ROOT/'control';root.mkdir(parents=True,exist_ok=True)
    lock=(root/'worker.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (root/'complete.json').exists():return
    runtime=Path(__file__).resolve().parents[3];python=sys.executable
    atomic(root/'worker.json',dict(pid=os.getpid(),host=socket.gethostname(),job=os.environ['SLURM_JOB_ID'],CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],runtime=str(runtime),started=time.time()),replace=True)
    def step(name,argv):
        atomic(root/'progress.json',dict(phase=name,started=time.time(),command=argv),replace=True)
        with (root/(name+'.log')).open('ab',buffering=0) as log:
            p=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,cwd=runtime);rc=p.wait()
        atomic(root/(name+'-exit.json'),dict(exit_code=rc,checked_unix=time.time()),replace=True)
        if rc:raise RuntimeError(name+' failed with exit '+str(rc))
    module=lambda n:[python,'-B','-m',n]
    try:
        normal=ROOT/'normal'
        if not (normal/'complete.json').exists():step('export',module('reproduction.grounding_eval.operations.dynrpo_export_normal')+['--base',BASE,'--data',DATA,'--output',str(normal)])
        receipt=read(normal/'complete.json');assert receipt['count']==3610 and receipt['correct']==1706
        reports={}
        for name,mod,verifier in [('v4','reproduction.grounding_eval','verify_completed_cf.py'),('v5','reproduction.grounding_eval.strict_v5','verify_completed_strict_v5.py')]:
            out=ROOT/('cf-'+name);manifest=out/'manifest.json';cmd=module(mod)
            if not manifest.exists():step(name+'-prepare',cmd+['prepare','--source',str(normal/'normal.jsonl'),'--normal-config',str(normal/'normal-config.json'),'--checkpoint',receipt['model'],'--output',str(out),'--method','DynRPO','--run-id',RUN,'--expected-count','3610'])
            if not (out/'verified-summary.json').exists():
                step(name+'-pilot',cmd+['pilot','--manifest',str(manifest),'--max-batch','256','--max-memory-utilization','0.85'])
                step(name+'-run',cmd+['run','--manifest',str(manifest),'--max-batch','256','--max-memory-utilization','0.85'])
                step(name+'-verify',cmd+['verify','--manifest',str(manifest)])
            acceptance=out/'independent-acceptance.json'
            if not acceptance.exists():step(name+'-independent',[python,'-B',str(runtime/'reproduction/grounding_eval/operations'/verifier),'--manifest',str(manifest),'--runtime',str(runtime),'--output',str(acceptance)])
            replay=out/'changed-rule-replay.json'
            if not replay.exists():step(name+'-rule-replay',module('reproduction.grounding_eval.operations.official_changed_rule_replay')+['--method','DynRPO','--cf-root',str(out),'--output',str(replay)])
            reports[name]=dict(summary=read(out/'verified-summary.json'),independent_acceptance=read(acceptance),changed_rule=read(replay),manifest_sha256=sha(manifest))
        atomic(root/'complete.json',dict(status='DYNRPO_NQ_GROUNDING_COMPLETE',normal=receipt,protocols=reports,checked_unix=time.time(),training_modified=False))
        atomic(root/'progress.json',dict(phase='COMPLETE',checked_unix=time.time()),replace=True)
    except BaseException as e:
        atomic(root/'failure.json',dict(error=repr(e),checked_unix=time.time()),replace=True);raise
if __name__=='__main__':main()
