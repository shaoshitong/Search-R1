"""Start the one new-job archive controller; credentials remain in RAM only."""
import importlib.util
import json
from pathlib import Path
import re

W = Path(__file__).resolve().parents[4]
OUT = W.parents[1]/'outputs/easy2560-co-rollout-hpc3-4gpu-20261017'
ROOT = Path(__file__).resolve().parent
COMMIT = '4cc07f3103917f5ac6bb81f6b022d454809c1054'
entry = W/'reproduction/easy2560_joint4/code_backup/recovery5-after730912-hk-source-8033-20261018/deploy_exact8033_HK_source_once.py'
spec = importlib.util.spec_from_file_location('existing_single_transport', entry)
transport = importlib.util.module_from_spec(spec); spec.loader.exec_module(transport)
secret = Path('/Users/shaoshitong/Documents/remote_codex_control/memories/hpc3-credentials.private.md').read_text()
line = next(x for x in secret.splitlines() if x.startswith('账号 jxu957'))
password = re.split(r'密码\s*[:：]?\s*', line, 1)[1].strip().strip('`')
secret = line = None
payload = dict(password=password, job='734689',
    base='/data/user/jxu957/experiments/easy2560-corollout-joint4-micro8-after731063-20261018',
    source_commit=COMMIT, run_id='dfe9c94084bb48b2a9bb7b60f9d882da',
    source_root='/backup/shaoshitong/easy2560-corollout-hpc3-4gpu-20261017/source/'+COMMIT)
result = transport.run_ssh((ROOT/'launch-734689-HK-archive-controller.py').read_text(),
    payload, OUT/'micro8-734689-HK-archive-launch-once.json')
password = payload = None
print(json.dumps(result))
