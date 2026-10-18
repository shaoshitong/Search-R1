"""Bounded foreground read-only follow-up; does not schedule or control training."""
import argparse
import json
from pathlib import Path
import runpy
import shlex
import subprocess
import time
from reproduction.hpc3_prefix5.direct_control import run
from reproduction.hpc3_prefix5.control import PYTHON,OUTPUT
from reproduction.hpc3_prefix5.observe import query_for


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--samples',type=int,default=80);parser.add_argument('--interval',type=int,default=30);args=parser.parse_args()
    assert 1<=args.samples<=160 and 10<=args.interval<=60
    operator=runpy.run_path(str(Path(__file__).with_name('hpc3-progressive-observe-20261018.py')))
    code=query_for(operator['BASE']).replace('print(json.dumps(result,allow_nan=False))',operator['EXTRA']+'\nprint(json.dumps(result,allow_nan=False))')
    code=code.replace("result['shared_free_bytes']=", "result['bootstrap_complete']=bool((b/'preparation/progressive-bootstrap-complete.json').exists())\nresult['shared_free_bytes']=")
    path=OUTPUT/'progressive-cp50-follow-five.jsonl'
    def counts(rows):return [None if r is None else r.get('completed_trajectories') for r in rows]
    for ordinal in range(args.samples):
        try:
            address=subprocess.check_output(['ipconfig','getifaddr','en0'],text=True).strip()
            r=run(PYTHON+' -c '+shlex.quote(code),account='mou176',source_address=address,timeout=25)
            if r.returncode:raise RuntimeError(r.stdout.decode()[-1200:]+r.stderr.decode()[-400:])
            v=json.loads(r.stdout);(OUTPUT/'progressive-cp50-current.json').write_text(json.dumps(v,indent=2)+'\n')
            row=dict(sample=ordinal+1,epoch=v['epoch'],job=v['job'],queue=v['queue'],stage=v.get('status'),
                     complete_outer=v['complete_outer'],bootstrap_complete=v['bootstrap_complete'],actor=v['actor'],probe=v['probe'],
                     prefix=counts(v.get('prefix',[])),full=counts(v.get('full',[])),
                     refresh={n:dict(prefix=counts(x['prefix']),full=counts(x['full-pool']),diagnostic_full=counts(x['diagnostic']['full'])) for n,x in v.get('probe_refreshes',{}).items()},
                     fit={n:dict(progress=[r.get('actual_adam_steps') if r else None for r in x['progress']],
                                  epochs=[r['epoch'] for r in x['epoch_metrics'] if r],complete=x['complete']) for n,x in v.get('probe_fits',{}).items()},
                     failed_jobs=v.get('failed_jobs'),errors=v.get('error_lines'),mixed=v.get('latest_selected'),rank_updates=v.get('inflight_rank_updates'))
            with path.open('a') as handle:handle.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
            if not v['queue'] or v['failed_jobs']:
                print(json.dumps(dict(status='ATTENTION_TERMINAL_OR_FAILED_PHASE',logs=v.get('recent_logs'))),flush=True);return
            if v['complete_outer']>=55 and v['status']['outer']>=56 and any((x or 0)>0 for x in counts(v.get('prefix',[]))):
                print(json.dumps(dict(status='FIVE_COMPLETE_RESUMED_OUTERS_AND_POSITIVE_NEXT_PHASE',outer=v['complete_outer'])),flush=True);return
        except Exception as error:
            print(json.dumps(dict(sample=ordinal+1,status='READ_FAILED_NOT_CURRENT_PROGRESS',error=repr(error))),flush=True)
        if ordinal+1<args.samples:time.sleep(args.interval)
    print(json.dumps(dict(status='BOUNDED_READ_ONLY_FOLLOWUP_EXPIRED_NOT_TRAINING_TERMINAL')),flush=True)

if __name__=='__main__':main()
