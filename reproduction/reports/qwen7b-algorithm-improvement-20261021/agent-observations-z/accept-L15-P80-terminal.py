"""Invoke the already staged CPU-only terminal validator after actual completion."""
import json
from pathlib import Path
import subprocess

out = Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-mixed-P80-terminal-CPU-acceptance-20261023-v1')
accepted = out / 'layer15-prefix80/terminal-accepted.json'
if accepted.exists():
    value = json.loads(accepted.read_bytes())
    assert value['terminal_gate']['job_id'] == '749782'
    print(json.dumps({'reused_existing_terminal_acceptance': True, 'actual_acceptance': value}))
else:
    command = ['bash', str(out / 'code-b3a3917b4d7a/acceptance.sh'), '--layer', '15', '--source-sha', '0a38f57a27f096b1bc1881a89bda121642f49eccbf3c45249b68508fd66b58a4']
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=120)
    if result.returncode:
        raise RuntimeError(result.stdout[-1400:] + result.stderr[-1400:])
    print(result.stdout)
