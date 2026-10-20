"""Copy only small immutable code/config/receipts; never model or raw outputs."""
import hashlib, importlib.util, json, shlex, subprocess
from pathlib import Path
C=Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5')
BASE='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/g20-exact-uniform15504-dp-v1'
NAMES=('config.json','source-publication.json','preparation.json','submission.json','train.sbatch','next-update-replay.json','independent-cp-000001.json')
spec=importlib.util.spec_from_file_location('backup_dp_ctl',C/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl)
address=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip()
code='''import hashlib,json,pathlib,time
b=pathlib.Path(BASE);result={}
for name in NAMES:
 p=b/name;raw=p.read_bytes();assert len(raw)<128*1024 and p.suffix in ('.json','.sbatch')
 result[name]=dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),text=raw.decode('utf8'))
print(json.dumps(dict(epoch=time.time(),files=result,model_checkpoint_feature_or_raw_prediction_copied=False)))
'''
command='python3 -c '+shlex.quote('BASE='+repr(BASE)+'\nNAMES='+repr(NAMES)+'\n'+code);assert len(command.encode())<16000
done=ctl.run(command,account='mou176',source_address=address,timeout=55)
if done.returncode:raise RuntimeError(done.stderr.decode()[-1200:])
value=json.loads(done.stdout.decode().splitlines()[-1]);out=Path(__file__).parent/'actual-runtime-metadata';out.mkdir(exist_ok=True)
for name,entry in value['files'].items():
 raw=entry.pop('text').encode();assert len(raw)==entry['bytes'] and hashlib.sha256(raw).hexdigest()==entry['sha256'];path=out/name
 if path.exists():assert path.read_bytes()==raw,'Immutable metadata changed'
 else:path.write_bytes(raw)
(out/'backup-manifest.json').write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps(dict(status='SMALL_ACTUAL_CONFIG_CODE_RECEIPTS_BYTE_IDENTICAL_BACKUP',files=len(value['files']),bytes=sum(x['bytes'] for x in value['files'].values()),model_or_prediction_payload_copied=False)))
