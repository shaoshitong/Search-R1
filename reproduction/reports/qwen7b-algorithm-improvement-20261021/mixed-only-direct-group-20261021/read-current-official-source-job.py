"""Read only the small immutable production config/source/job bindings."""
import hashlib
import json
from pathlib import Path
import subprocess
import time
b = Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/joint-mixed-direct-official-169028-v1')
def proof(p):
    raw = p.read_bytes()
    return dict(path=str(p), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
def read(p): return json.loads(p.read_bytes())
config = read(b / 'config.json'); pub = read(b / 'source-publication.json'); result = read(b / 'submission-result.json')
assert result['job_id'] == '746083' and proof(b / 'config.json') == result['config']
assert config['initialization_source'] == 'official_initial' and config['official_revision'] == 'a09a35458c702b33eeacc393d103063234e8bc28'
assert config['loss'] == dict(binary_mixed_weight=1.) and config['initial_native_Adam_steps'] == config['initial_group_head_Adam_steps'] == 0
assert not config['old_probe_checkpoint_loaded'] and not config['old_probe_Adam_or_RNG_inherited']
code = Path(pub['code_root'])
for name, ref in pub['files'].items():
    actual = proof(code / name)
    assert actual['bytes'] == ref['bytes'] and actual['sha256'] == ref['sha256']
out = dict(epoch=time.time(), config=dict(reference=proof(b / 'config.json'), value=config),
    source_publication=dict(reference=proof(b / 'source-publication.json'), value=pub),
    entry=dict(reference=proof(b / 'training.sbatch'), text=(b / 'training.sbatch').read_text()),
    published_runtime_files_fresh_SHA_passed=True)
for name in ('controller.json', 'progress.json', 'evaluation-progress.json', 'first-update-replay.json', 'first-cp-independent-audit.json', 'durable-next-update-replay.json', 'latest.json', 'failure.json', 'complete.json'):
    p = b / name
    out[name] = read(p) if p.exists() else None
log = b / 'logs/train-746083.log'
if log.exists():
    with log.open('rb') as f:
        f.seek(max(0, log.stat().st_size - 3500))
        out['log_tail'] = f.read().decode('utf-8', 'replace')
out['actual_job'] = subprocess.check_output(['scontrol', 'show', 'job', '-o', '746083'], universal_newlines=True)
print(json.dumps(out, sort_keys=True))
