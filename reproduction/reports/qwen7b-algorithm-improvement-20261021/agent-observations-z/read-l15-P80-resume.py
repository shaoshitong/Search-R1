"""Read only the independently resumed L15/P80 job's small metadata."""
import json
import pathlib
import subprocess
import time

b = pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-mixed-layer-prefix-20261022-2gpu-v1/layer15-prefix80')
out = {'time': time.time(), 'base': str(b)}
for name in ('root-submission.json', 'controller.json', 'progress-rank-0.json', 'progress-rank-1.json', 'latest.json', 'complete.json'):
    p = b / name
    if p.exists():
        out[name] = json.loads(p.read_bytes())
job = str(out['root-submission.json']['job_id'])
assert job == '749782', job
out['current_job_id'] = job
for name, args in (
    ('accounting', ['sacct', '-X', '-n', '-P', '-j', job, '--format=JobIDRaw,State,ExitCode,Elapsed,NodeList']),
    ('queue', ['squeue', '-h', '-j', job, '-o', '%i|%j|%T|%M|%R']),
):
    p = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=15)
    out[name] = {'returncode': p.returncode, 'stdout': p.stdout.strip(), 'stderr': p.stderr.strip()}
out['failure_files'] = [p.name for p in b.glob('failure-rank-*.json')]
p = b / 'curve.jsonl'
rows = []
if p.exists():
    with p.open('rb') as f:
        f.seek(max(0, p.stat().st_size - 30000))
        lines = f.read().decode().splitlines()
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get('epoch') == 3 and row.get('cursor', 0) > 128:
            rows.append(row)
out['recent_E3_real_updates'] = rows[-10:]
for p in b.glob('evaluation-epoch-*.json'):
    e = json.loads(p.read_bytes())
    out[p.name] = e
print(json.dumps(out))
