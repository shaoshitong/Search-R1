"""Read stored pre-fit and full-set epoch metrics; never run extra inference."""
import argparse
import json
import shlex
import subprocess

from reproduction.hpc3_prefix5.control import OUTPUT, PYTHON
from reproduction.hpc3_prefix5.direct_control import run

QUERY = r'''
import hashlib,json,time
from pathlib import Path
b=Path('/data/user/jxu957/experiments/dynrpo-prefix5-thread5-20261017-r4');n=TARGET_OUTER
root=b/'outers'/('outer-%06d'%n);fit=root/'probe-update'
def rd(p):return json.loads(p.read_text())
if not (root/'complete.json').exists():
 print(json.dumps(dict(status='PENDING_NO_ACTUAL_P10_JOINT',outer=n)));raise SystemExit(0)
j=rd(root/'complete.json');previous=rd(b/'outers'/('outer-%06d'%(n-1))/'complete.json')
assert j['runtime_source_commit']=='2000d46a530c64d78be7802bf11175a68b65effe'
assert j['periodic_probe_update'] and j['outer']==n
raw=(fit/'metrics.json').read_bytes();m=json.loads(raw);done=rd(fit/'complete.json')
assert hashlib.sha256(raw).hexdigest()==done['metrics_sha256']
assert m['outer']==n and m['run_id']==j['run_id'] and m['actor_version']==j['probe_metrics']['actor_version']
assert m['epochs']==3 and m['new_adam_steps_per_rank']==150
assert m['actual_adam_steps_per_rank']==j['probe']['actual_adam_steps_per_rank']
before=previous['probe']['actual_adam_steps_per_rank'];assert len(set(before))==1
rows=[dict(epoch=0,adam_per_rank=before,metrics=m['before_update'])]
for epoch in (1,2,3):
 e=rd(fit/('epoch-%03d-metrics.json'%epoch));c=rd(fit/('epoch-%03d.json'%epoch))
 expected=[before[0]+50*epoch]*4
 assert e['epoch']==epoch and e['prefix_tokens']==m['prefix_tokens']
 assert e['actual_adam_steps_per_rank']==c['actual_adam_steps_per_rank']==expected
 for split,count in (('train',2560),('test',512)):
  assert e['metrics'][split]['count']==count
  assert e['metrics'][split]['true_mixed']=='0<sum(own_strict_em)<5'
 rows.append(dict(epoch=epoch,adam_per_rank=expected,metrics=e['metrics']))
assert rows[-1]['adam_per_rank']==m['actual_adam_steps_per_rank']
assert rows[-1]['metrics']=={k:m[k] for k in ('train','test')}
print(json.dumps(dict(status='ACTUAL_P%d_FIT_CURVE_SCALARS_READ'%n,checked_unix=time.time(),outer=n,
 actor_version=m['actor_version'],prefix_tokens=m['prefix_tokens'],epochs=3,train_questions=2560,diagnostic_questions=512,
 actual_adam_steps_per_rank=m['actual_adam_steps_per_rank'],train_loop_seconds=m['train_loop_seconds'],
 metrics_sha256=done['metrics_sha256'],rows=rows,formal_QA_EM=False,
 selected={k:j['rollout'][k] for k in ('actual_selected_mixed_questions','actual_selected_mixed_fraction','fallback_count')})))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outer', type=int, choices=range(70, 200, 10), required=True)
    args = parser.parse_args()
    address = subprocess.check_output(['ipconfig', 'getifaddr', 'en0'], text=True).strip()
    result = run(PYTHON + ' -c ' + shlex.quote(QUERY.replace('TARGET_OUTER', str(args.outer))),
                 account='mou176', source_address=address, timeout=25)
    if result.returncode:
        raise RuntimeError('Stored curve read failed: ' + result.stdout.decode()[-1000:] + result.stderr.decode()[-1000:])
    value = next(json.loads(line) for line in reversed(result.stdout.decode().splitlines())
                 if line.lstrip().startswith('{'))
    if 'rows' in value:
        (OUTPUT / ('two-stage-periodic-%06d-fit-curve.json' % args.outer)).write_text(json.dumps(value, indent=2)+'\n')
    print(json.dumps(value))


if __name__ == '__main__':
    main()
