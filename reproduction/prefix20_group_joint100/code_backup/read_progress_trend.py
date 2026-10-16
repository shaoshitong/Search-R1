import json, sys, hashlib
from pathlib import Path

b = Path('/mnt/data/shaoshitong/search-g1-question20-groupjoint100-20261016-cee20bcd')
sys.path.insert(0, str(b / 'repo'))
from reproduction.prefix20_group_joint100.launch import observe

x = observe(b)
latest = x['latest']
n = latest['epoch'] if latest else 0
rows = []
for epoch in range(1, n + 1):
    d = b / 'train/epochs' / ('epoch-%04d' % epoch)
    p = d / 'metrics.json'
    raw = p.read_bytes()
    m = json.loads(raw)
    c = json.loads((d / 'complete.json').read_text())
    assert c['epoch'] == epoch and c['actual_adam_steps_per_rank'] == [100 * epoch] * 8
    assert m['test_manifest_sha256'] == 'a125fc5af90e34fcd728c82077d9d62070a78d5ef703ac11f3748687d088e64f'
    rows.append(dict(
        epoch=epoch, source_data_epoch=m['source_data_epoch'], observed_utc=m['observed_utc'],
        actual_adam_steps_per_rank=m['actual_adam_steps_per_rank'], actual_lr=m['actual_learning_rate'],
        train=m['train_mixed_group'], test=m['test_mixed_group'],
        online_trajectory_bce=m['train_online_trajectory_bce'], online_group_bce=m['train_online_group_bce'],
        train_loop_seconds_including_inline_encoder=m['train_seconds'],
        train_diagnostic_seconds=m['train_diagnostic_seconds'], test_diagnostic_seconds=m['test_diagnostic_seconds'],
        complete_utc=c['observed_utc'], metrics_sha256=hashlib.sha256(raw).hexdigest()))
print(json.dumps(dict(observation=x, complete_rows=rows, all_large_outputs_stayed_remote=True)))
