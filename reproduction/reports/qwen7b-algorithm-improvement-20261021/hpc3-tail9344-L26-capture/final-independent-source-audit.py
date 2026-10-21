"""Runs only the published stdlib metadata/identity auditor remotely."""
import base64,hashlib,importlib.util,json,shlex,subprocess,zlib
from pathlib import Path
CONTROL=Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5')
REPO=CONTROL/'work/qwen7b-probe-warmup-20261019'
source=(REPO/'reproduction/operations/qwen7b_warmup/audit_hpc3_tail_l26_bank.py').read_bytes()
body="import sys,os;os.environ['AUDITOR_SOURCE_SHA256']="+repr(hashlib.sha256(source).hexdigest())+";sys.argv=['audit','--base','/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/hpc3-tail9344-independent-L26-prepared-v2','--job-id','745731','--questions','9344'];exec("+repr(source.decode())+")"
compile(body,'published-final-source-audit','exec')
spec=importlib.util.spec_from_file_location('fixed9344_final_audit_control',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl);ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0']).decode().strip();packed=base64.b64encode(zlib.compress(body.encode())).decode();cmd='python3 -c '+shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(packed)+')))');r=ctl.run(cmd,account='mou176',source_address=ip,timeout=55)
if r.returncode:raise RuntimeError(r.stdout[-2000:]+r.stderr[-2000:])
v=json.loads(r.stdout.splitlines()[-1]);Path(__file__).with_name('actual-final-independent-source-audit.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v))
