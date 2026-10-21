"""Read a <=16KB existing aggregate receipt; no payload/tensor/GPU reads."""
import hashlib,importlib.util,json,pathlib,shlex,subprocess
ROOT=pathlib.Path(__file__).resolve().parents[4];CONTROL=ROOT.parents[1];OUT=pathlib.Path(__file__).parent
p='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/direct-hidden-count30016-train-v2/evaluations/cursor-000938/terminal-inherited-independent-audit.json'
code="import pathlib,hashlib,json;p=pathlib.Path("+repr(p)+");assert p.stat().st_size==4952;data=p.read_bytes();assert hashlib.sha256(data).hexdigest()=='6a8b730441f23b1acbc69b67bc1d52e246ec03999d312d14730f199b1ddb00e4';print(data.decode())"
spec=importlib.util.spec_from_file_location('final_small_receipt_control',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl)
r=ctl.run('python3 -c '+shlex.quote(code),account='mou176',source_address=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0']).decode().strip(),timeout=30)
if r.returncode:raise RuntimeError(r.stderr[-1000:])
v=json.loads(r.stdout);(OUT/'actual-cursor-000938-terminal-inherited-full-receipt.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(dict(status=v['status'],host=v['host'],mode=v['audit_mode'],complete=v['producer_complete'],evaluation=v['evaluation_receipt'],predictions=v['predictions'])))
