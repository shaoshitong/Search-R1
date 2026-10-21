"""Read back only small actual code/config/Slurm receipts; never data payloads."""
import base64,hashlib,importlib.util,json,shlex,subprocess,zlib
from pathlib import Path
CONTROL=Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5')
BODY='''import pathlib,json,base64,hashlib
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/joint-balanced-native-E4-124194-v1');out={}
for name in('config.json','training.sbatch','source-publication.json','preparation.json','submission-identity.json','submission-result.json'):
 p=b/name;raw=p.read_bytes();assert len(raw)<20000;out[name]=dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),base64=base64.b64encode(raw).decode())
print(json.dumps(out))
'''
spec=importlib.util.spec_from_file_location('backup_native_main_control_only',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl);ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0']).decode().strip();packed=base64.b64encode(zlib.compress(BODY.encode())).decode();cmd='python3 -c '+shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(packed)+')))');r=ctl.run(cmd,account='mou176',source_address=ip,timeout=30)
if r.returncode:raise RuntimeError(r.stdout[-3000:]+r.stderr[-1000:])
v=json.loads(r.stdout.splitlines()[-1]);root=Path(__file__).parent/'actual-control';root.mkdir(exist_ok=True);out={}
for name,value in v.items():
 raw=base64.b64decode(value['base64']);assert len(raw)==value['bytes']and hashlib.sha256(raw).hexdigest()==value['sha256'];p=root/name
 if p.exists():assert p.read_bytes()==raw
 else:p.write_bytes(raw)
 out[name]={k:value[k]for k in('path','bytes','sha256')}
Path(__file__).with_name('actual-control-readback.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
