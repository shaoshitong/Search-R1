"""Read only the exact original MAE arm and small progress/evaluation metadata."""
import json
from pathlib import Path
import subprocess
import time

suite = Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-packed-full169028-M3to10-P20-20261022-v1')
base = suite / 'mae'
submission = json.loads((base / 'root-submission.json').read_bytes())
assert str(submission['job_id']) == '749569'
assert submission['publication_commit'] == '7ac274ac8ff5dea0a12dfd44cfb03fe57491458f'
out = {'time': time.time(), 'job_id': '749569', 'arm': 'mae', 'files': {}, 'final4096_read': False}
for key in ('job_id', 'name', 'epochs', 'EMA_decay', 'loss', 'publication_commit', 'time'):
    out.setdefault('submission', {})[key] = submission[key]
for name in ('controller.json', 'progress-rank-0.json', 'progress-rank-1.json',
             'evaluation-progress-rank-0.json', 'evaluation-progress-rank-1.json',
             'latest.json', 'complete.json'):
    path = base / name
    if path.exists():
        out['files'][name] = json.loads(path.read_bytes())
for pattern in ('failure-rank-*.json', 'coverage-epoch-*.json', 'evaluation-epoch-*.json'):
    for path in sorted(base.glob(pattern)):
        out['files'][path.name] = json.loads(path.read_bytes())
for key, command in [('accounting', ['sacct', '-X', '-n', '-P', '-j', '749569', '-o', 'JobIDRaw,State,ExitCode,Elapsed,NodeList']),
                     ('queue', ['squeue', '--start', '-h', '-j', '749569', '-o', '%i|%j|%P|%T|%S|%R'])]:
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=15)
    out[key] = {'code': result.returncode, 'stdout': result.stdout.strip(), 'stderr': result.stderr.strip()}
log = base / 'train-749569.log'
if log.exists():
    with log.open('rb') as stream:
        stream.seek(max(0, log.stat().st_size - 2000))
        out['log_tail'] = stream.read().decode(errors='replace')
print(json.dumps(out))
