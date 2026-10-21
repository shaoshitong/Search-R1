"""Back up only the immutable staged code-origin metadata, never model/data."""
import base64,hashlib,importlib.util,json,shlex,subprocess,zlib
from pathlib import Path
CONTROL=Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5')
body="""import pathlib,json,hashlib,base64,time
p=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/joint-mixed-direct-official-169028-v1/source-publication.json');raw=p.read_bytes();assert len(raw)==6458 and hashlib.sha256(raw).hexdigest()=='cdae3137aec68c9521afe2bb11e60fb67fc5922b7db00bb076b0e2f787bfce80';print(json.dumps(dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),base64=base64.b64encode(raw).decode(),readback_epoch=time.time())))
"""
spec=importlib.util.spec_from_file_location('official_staged_publication_backup',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl);ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0']).decode().strip();packed=base64.b64encode(zlib.compress(body.encode())).decode();cmd='python3 -c '+shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(packed)+')))');r=ctl.run(cmd,account='mou176',source_address=ip,timeout=25)
if r.returncode:raise RuntimeError(r.stdout[-1000:]+r.stderr[-1000:])
x=json.loads(r.stdout.splitlines()[-1]);raw=base64.b64decode(x.pop('base64'));assert len(raw)==x['bytes']and hashlib.sha256(raw).hexdigest()==x['sha256'];b=Path(__file__).parent;(b/'actual-source-publication.json').write_bytes(raw);(b/'actual-source-publication-readback.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x))
