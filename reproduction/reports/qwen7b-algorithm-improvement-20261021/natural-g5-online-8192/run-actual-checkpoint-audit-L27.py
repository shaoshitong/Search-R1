"""Remote CPU-only audit entry; select one finished original layer at a time."""
import json
import os
from pathlib import Path
import subprocess

LAYER = 27
EPOCH = 3
base = Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-natural-g5-8192-20261022-v1')
publication = json.loads((base / 'source-publication.json').read_text())
runtime = Path(publication['code_root']) / 'reproduction/operations/qwen7b_warmup/runtime.sh'
script = base / 'postrun-audit-code-c91222bd3215/audit_natural_g5_online_checkpoint.py'
env = dict(os.environ, QWEN7B_CODE_ROOT=publication['code_root'], CUDA_VISIBLE_DEVICES='',
           OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
result = subprocess.run(['bash', str(runtime), '-u', str(script), '--layer', str(LAYER),
                         '--epoch', str(EPOCH)], env=env, stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE, universal_newlines=True, timeout=180)
if result.returncode:
    raise RuntimeError(result.stderr[-4000:] + result.stdout[-4000:])
print(result.stdout, end='')
