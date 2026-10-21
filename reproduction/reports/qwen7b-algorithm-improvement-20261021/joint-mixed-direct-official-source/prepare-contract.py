"""Runs the published stdlib official source/eval contract preparer remotely."""
import base64,importlib.util,json,shlex,subprocess,zlib
from pathlib import Path
CONTROL=Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5');REPO=CONTROL/'work/qwen7b-probe-warmup-20261019'
source=(REPO/'reproduction/operations/qwen7b_warmup/prepare_official_mixed_source_eval.py').read_text();compile(source,'official-source-eval-contract','exec');packed=base64.b64encode(zlib.compress(source.encode())).decode();command='python3 -c '+shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(packed)+')))');assert len(command.encode())<16000
spec=importlib.util.spec_from_file_location('official_fresh_source_private_control',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl);ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0']).decode().strip();r=ctl.run(command,account='mou176',source_address=ip,timeout=55)
if r.returncode:raise RuntimeError(r.stdout[-1500:]+r.stderr[-2500:])
x=json.loads(r.stdout.splitlines()[-1]);root=Path(__file__).parent;(root/'actual-contract-preparation.json').write_text(json.dumps(x,indent=2)+'\n');(root/'OFFICIAL-SOURCE-AND-EVAL-CONTRACT.json').write_text(json.dumps(x['value'],indent=2)+'\n');print(json.dumps(x,sort_keys=True))
