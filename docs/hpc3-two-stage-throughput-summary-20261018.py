"""Read committed R4 phase timings and authentic selected-G5 composition.

Only small JSON scalars cross SSH. This never opens rollout/tensor payloads,
changes training, or treats probe classification accuracy as formal QA EM.
"""
import json
import shlex
import subprocess

from reproduction.hpc3_prefix5.control import OUTPUT, PYTHON
from reproduction.hpc3_prefix5.direct_control import run

QUERY = r'''
import hashlib,json,time
from pathlib import Path
b=Path('/data/user/jxu957/experiments/dynrpo-prefix5-thread5-20261017-r4')
runtime='2000d46a530c64d78be7802bf11175a68b65effe'
config=json.loads((b/'preparation/launch-config.json').read_text())
assert json.loads((b/'preparation/slurm-submission.json').read_text())['job']=='733898'
assert json.loads((b/'preparation/source-publication.json').read_text())['commit']==runtime
rows=[]
previous=json.loads((b/'outers/outer-000061/complete.json').read_text())
for n in range(62,201):
 p=b/'outers'/('outer-%06d'%n)/'complete.json'
 if not p.exists():break
 raw=p.read_bytes();j=json.loads(raw)
 assert j['outer']==n and j['runtime_source_commit']==runtime
 assert j['run_id']==config['run_id'] and j['execution_id']==config['execution_id']
 r=j['rollout'];h={int(k):int(v) for k,v in r['selected_correct_count_histogram'].items()}
 assert set(h)==set(range(6)) and sum(h.values())==512 and r['trajectories']==2560
 mixed=sum(h[k] for k in range(1,5))
 assert mixed==r['actual_selected_mixed_questions'] and mixed/512==r['actual_selected_mixed_fraction']
 phases={k:float(v['elapsed_seconds']) for k,v in j['phase_timings'].items()}
 assert all(v['exit_code']==0 for v in j['phase_timings'].values())
 cycle=j['observed_unix']-previous['observed_unix']
 assert cycle>0
 rows.append(dict(outer=n,committed_unix=j['observed_unix'],joint_sha256=hashlib.sha256(raw).hexdigest(),
  prefix_tokens=r['head_prefix_tokens'],periodic_probe_update=j['periodic_probe_update'],
  selected_k_histogram={str(k):h[k] for k in range(6)},all_wrong_fraction=h[0]/512,
  mixed_fraction=mixed/512,all_correct_fraction=h[5]/512,fallback_count=r['fallback_count'],
  phase_seconds=phases,successful_phase_seconds_sum=sum(phases.values()),
  joint_to_joint_seconds=cycle,joint_to_joint_includes_storage_and_orchestration=True,
  actor_actual_adam=j['actor']['actual_adam_steps_per_rank'],actor_new_adam=j['actor']['new_adam_steps_per_rank'],
  actor_skipped_opportunities=j['actor']['skipped_update_opportunities'],
  probe_actual_adam=j['probe']['actual_adam_steps_per_rank']))
 previous=j
print(json.dumps(dict(status='COMMITTED_R4_PHASE_TIMING_AND_STORED_STRICT_G5_COMPOSITION',checked_unix=time.time(),
 job='733898',runtime_source_commit=runtime,run_id=config['run_id'],rows=rows,
 formal_QA_EM=False,score_recomputed=False,training_modified=False)))
'''


def main():
    address = subprocess.check_output(['ipconfig', 'getifaddr', 'en0'], text=True).strip()
    result = run(PYTHON + ' -c ' + shlex.quote(QUERY), account='mou176',
                 source_address=address, timeout=25)
    if result.returncode:
        raise RuntimeError('Scalar observation failed: ' + result.stderr.decode()[-1000:])
    value = next(json.loads(line) for line in reversed(result.stdout.decode().splitlines())
                 if line.lstrip().startswith('{'))
    path = OUTPUT / 'two-stage-r4-throughput-and-g5-composition.json'
    path.write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(dict(status=value['status'],count=len(value['rows']),
                          latest=value['rows'][-1] if value['rows'] else None,
                          summary_path=str(path))))


if __name__ == '__main__':
    main()
