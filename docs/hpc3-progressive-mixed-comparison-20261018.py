"""Read completed paired outers; retrieve scalar aggregates only."""
import json
import shlex
import subprocess
from reproduction.hpc3_prefix5.direct_control import run
from reproduction.hpc3_prefix5.control import PYTHON, OUTPUT

QUERY = r'''
import json,time,hashlib
from pathlib import Path
root=Path('/data/user/jxu957/experiments')
bases=[root/'dynrpo-prefix5-thread5-20261017-r2',root/'dynrpo-prefix5-thread5-20261017-r3']
def read(p):return json.loads(p.read_text())
def row(base,outer):
 p=base/'outers'/('outer-%06d'%outer)/'complete.json'
 if not p.exists():return None
 j=read(p);r=j['rollout'];budget=read(p.parent/'candidate-budget.json')
 return dict(candidates=budget['questions'],mixed=r['actual_selected_mixed_questions'],
  fraction=r['actual_selected_mixed_fraction'],selected_questions=r['questions'],
  probe_adam=j['probe']['actual_adam_steps_per_rank'][0],sampled_actor=r['actor_version'],
  actor_after=j['actor']['actual_adam_steps_per_rank'],joint_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
rows=[dict(outer=n,old_prefix5_epoch1=row(bases[0],n),new_prefix20_epoch3=row(bases[1],n)) for n in range(51,56)]
paired=[r for r in rows if all(r[k] is not None for k in ('old_prefix5_epoch1','new_prefix20_epoch3'))]
summary={}
for k in ('old_prefix5_epoch1','new_prefix20_epoch3'):
 summary[k]=dict(rounds=len(paired),mixed=sum(r[k]['mixed'] for r in paired),questions=512*len(paired),
  mean_fraction=sum(r[k]['fraction'] for r in paired)/len(paired) if paired else None)
print(json.dumps(dict(status='ACTUAL_MATCHED_COMPLETE_OUTER_MIXED_COMPARISON',rows=rows,summary=summary,
 checked_unix=time.time(),changed_together=['prefix5_to20','CP50_three_epoch_refit','candidate_cap20480_to5120'],
 subsequent_actor_trajectories_diverge=True,isolated_prefix_causal_effect_claimed=False)))
'''

def main():
    address = subprocess.check_output(['ipconfig', 'getifaddr', 'en0'], text=True).strip()
    result = run(PYTHON + ' -c ' + shlex.quote(QUERY), account='mou176',
                 source_address=address, timeout=25)
    if result.returncode:
        raise RuntimeError(result.stdout.decode()[-2000:] + result.stderr.decode()[-1000:])
    value = next(json.loads(line) for line in reversed(result.stdout.decode().splitlines())
                 if line.lstrip().startswith('{'))
    (OUTPUT / 'progressive-vs-old-mixed-latest-20261018.json').write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(value))

if __name__ == '__main__':
    main()
