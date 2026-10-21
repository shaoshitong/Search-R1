"""Read only sealed metadata; never bulk cache or model/checkpoint tensors."""
import importlib.util,json,pathlib,shlex,subprocess
R=pathlib.Path(__file__).resolve().parents[4];C=R.parents[1];HERE=pathlib.Path(__file__).parent
BASE='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/direct-count-frozenE1-fit30016-hold5120-v2'
CSHA='fb637c02812dd070cf392a58df75e0d219bbb10d1a2d24cd6e201f3638147767'
spec=importlib.util.spec_from_file_location('largecap_metadata_private',C/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl)
body="""import pathlib,json,hashlib,time
b=pathlib.Path(BASE)
def read(p):return json.loads(p.read_bytes())
def proof(p):
 raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
assert proof(b/'config.json')['sha256']==CSHA
out=dict(epoch=time.time(),target=str(b),config=proof(b/'config.json'))
p=b/'cache/complete.json'
if p.exists():
 c=read(p);out['cache_complete']=dict(file=proof(p),status=c['status'],config_sha256=c['config_sha256'],trajectory_ids_sha256=c['trajectory_ids_sha256'],files=c['files'],chunks=len(c['chunks']),frozen_legacy_before=c['frozen_legacy_before'],frozen_legacy_after=c['frozen_legacy_after'],actual_native_hidden_dtype=c['actual_native_hidden_dtype'],same_api_golden=c['same_api_golden'],actor_updates=c['actor_updates'],legacy_updates=c['legacy_updates'])
for name in ('latest.json','progress.json','first-update-replay.json','failure.json','complete.json'):
 p=b/name
 if p.exists():out[name[:-5]]=read(p)
for cursor in range(7):
 p=b/'checkpoints'/('cursor-%06d.pt'%cursor)
 if p.exists():out['CP%d_actual_stat'%cursor]=dict(path=str(p),bytes=p.stat().st_size)
for epoch in(0,1,2,4,8,20):
 p=b/'evaluations'/('epoch-%02d'%epoch)/'complete.json'
 if p.exists():
  n=read(p);cp=n['checkpoint'];assert 0<cp['bytes']<64*1024**2 and pathlib.Path(cp['path']).stat().st_size==cp['bytes'];actual=proof(pathlib.Path(cp['path']));assert all(actual[k]==cp[k]for k in('path','bytes','sha256'));out['epoch%d_complete'%epoch]=dict(file=proof(p),epoch=n['epoch'],cursor=n['cursor'],checkpoint=cp,checkpoint_fresh_file_SHA256_verified=True,checkpoint_tensor_load_performed=False,metrics=n['metrics'],training_components_unchanged=n['training_components_unchanged'])
print(json.dumps(out))"""
cmd='python3 -c '+shlex.quote('BASE='+repr(BASE)+';CSHA='+repr(CSHA)+'\n'+body)
assert len(cmd.encode())<16000
x=ctl.run(cmd,account='mou176',source_address=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0']).decode().strip(),timeout=35)
if x.returncode:raise RuntimeError('Read-only sealed metadata unavailable; no training-state inference')
out=json.loads(x.stdout.decode().splitlines()[-1]);(HERE/'actual-cache-and-head-state.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
