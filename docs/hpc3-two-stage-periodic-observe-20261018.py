"""Bounded scalar observation of a progressive P10's full-pool and fit phases."""
import argparse
import json
import shlex
import subprocess
from reproduction.hpc3_prefix5.direct_control import run
from reproduction.hpc3_prefix5.control import OUTPUT, PYTHON

QUERY = r'''
import json,time,subprocess
from pathlib import Path
b=Path('/data/user/jxu957/experiments/dynrpo-prefix5-thread5-20261017-r4');outer=OUTER
def rd(p,limit=1024**2):
 try:
  with p.open('rb') as f:raw=f.read(limit+1)
 except FileNotFoundError:return None
 assert len(raw)<=limit,str(p)
 return json.loads(raw)
c=rd(b/'preparation/launch-config.json');pub=rd(b/'preparation/source-publication.json')
job=rd(b/'preparation/slurm-submission.json')['job']
assert job=='733898' and pub['commit']=='2000d46a530c64d78be7802bf11175a68b65effe'
root=b/'outers'/('outer-%06d'%outer)
diagnostic=root/'diagnostic/outers'/('outer-%06d'%outer)
def counters(parent):
 result=[]
 for rank in range(4):
  value=rd(parent/('rank-%d-progress.json'%rank))
  result.append({key:value.get(key) for key in ('stage','completed_trajectories','expected_trajectories','actual_sampled_tokens','elapsed_seconds')} if value else None)
 return result
actor=rd(b/'actor/latest.json',128*1024**2);probe=rd(b/'probe/latest.json')
fitted=rd(root/'probe-update/complete.json')
value=dict(epoch=time.time(),job=job,queue=subprocess.check_output(['squeue','-h','-j',job,'-o','%i|%T|%P|%R'],text=True).strip(),
 status=rd(b/'logs/pipeline-status.json'),budget=rd(root/'candidate-budget.json'),
 prefix=counters(root/'prefix'),full_pool=counters(root/'full-pool'),selected_full=counters(root/'full'),
 diagnostic_prefix=counters(diagnostic/'prefix'),diagnostic_full=counters(diagnostic/'full'),
 actor={k:actor.get(k) for k in ('completed_outer','actual_adam_steps_per_rank','scheduler_completed_outer','skipped_update_opportunities')},
 probe={k:probe.get(k) for k in ('completed_outer','actual_adam_steps_per_rank','sha256')},
 fit=[rd(root/'probe-update'/('progress-rank-%d.json'%rank)) for rank in range(4)],
 epochs=[rd(root/'probe-update'/('epoch-%03d.json'%epoch)) for epoch in (1,2,3)],
 fitted={key:fitted.get(key) for key in ('status','outer','actual_adam_steps_per_rank','new_adam_steps_per_rank','epochs','prefix_tokens','actor_version')} if fitted else None,
 joint_complete=(root/'complete.json').is_file())
print(json.dumps(value))
'''

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outer', type=int, choices=range(60, 200, 10), required=True)
    args = parser.parse_args()
    address = subprocess.check_output(['ipconfig', 'getifaddr', 'en0'], text=True).strip()
    result = run(PYTHON + ' -c ' + shlex.quote(QUERY.replace('OUTER', str(args.outer))),
                 account='mou176', source_address=address, timeout=25)
    if result.returncode:
        raise RuntimeError('Read failed; training terminal is not inferred: ' + result.stderr.decode()[-1200:])
    value = next(json.loads(line) for line in reversed(result.stdout.decode().splitlines())
                 if line.lstrip().startswith('{'))
    path = OUTPUT / ('two-stage-periodic-%06d-current.json' % args.outer)
    path.write_text(json.dumps(value, indent=2) + '\n')
    with (OUTPUT / ('two-stage-periodic-%06d-observations.jsonl' % args.outer)).open('a') as f:
        f.write(json.dumps(value) + '\n')
    # Epoch receipts, paths and fit records remain in the small local audit;
    # terminal output only displays measured counters.
    print(json.dumps(dict(epoch=value['epoch'], queue=value['queue'], stage=value['status'],
        questions=value['budget']['questions'] if value['budget'] else None,
        **{key:[r['completed_trajectories'] if r else None for r in value[key]]
           for key in ('prefix','full_pool','diagnostic_prefix','diagnostic_full')},
        actor=value['actor'], probe=value['probe'],
        fit=[r['actual_adam_steps'] if r else None for r in value['fit']],
        epochs=[r['completed_epoch'] for r in value['epochs'] if r],
        fitted=value['fitted'],joint_complete=value['joint_complete'])))

if __name__ == '__main__':
    main()
