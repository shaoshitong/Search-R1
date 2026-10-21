"""Copy small nonsecret actual entry/config receipts only; never payload/model."""
import base64,hashlib,importlib.util,json,shlex,subprocess,zlib
from pathlib import Path
CONTROL=Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5')
BODY='''import pathlib,json,hashlib
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/hpc3-tail6818-independent-L26-prepared-v1');names=['source-publication.json','preparation.json','capture.sbatch','submission-plan.json','submission-intent.json','submission-identity.json','submission-result.json','pre-submit-physical-readback.json'];out={}
for name in names:
 p=b/name;raw=p.read_bytes();assert len(raw)<40000;out[name]=dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),text=raw.decode())
print(json.dumps(out))
'''
spec=importlib.util.spec_from_file_location('small6818backup',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl);ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0']).decode().strip();packed=base64.b64encode(zlib.compress(BODY.encode())).decode();cmd='python3 -c '+shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(packed)+')))');r=ctl.run(cmd,account='mou176',source_address=ip,timeout=20)
if r.returncode:raise RuntimeError(r.stdout[-1000:]+r.stderr[-1000:])
v=json.loads(r.stdout.splitlines()[-1]);root=Path(__file__).parent;dest=root/'actual-control';dest.mkdir(exist_ok=True);refs={}
for name,item in v.items():
 raw=item['text'].encode();assert len(raw)==item['bytes']and hashlib.sha256(raw).hexdigest()==item['sha256'];(dest/name).write_bytes(raw);refs[name]={k:item[k]for k in('path','bytes','sha256')}
(root/'actual-control-readback.json').write_text(json.dumps(refs,indent=2)+'\n');print(json.dumps(refs))
