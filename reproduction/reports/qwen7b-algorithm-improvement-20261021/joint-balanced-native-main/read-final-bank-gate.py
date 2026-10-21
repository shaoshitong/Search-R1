"""Read-only final124194 exact-binding gate, without payload/tensor loading."""
import base64,importlib.util,json,shlex,subprocess,zlib
from pathlib import Path
CONTROL=Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5')
BODY='''import pathlib,json,hashlib,time
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/joint124194-native-training-bank-v1')
def read(p):return json.loads(pathlib.Path(p).read_bytes())
def proof(p):
 p=pathlib.Path(p);raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
p=b/'complete.json'
if not p.exists():
 out=dict(status='WAITING_FOR_ROOT_FINAL_JOIN_COMPLETION',epoch=time.time(),actual_complete_exists=False,GPU_submitted=False)
 if(b/'progress.json').exists():out['root_progress']=read(b/'progress.json')
else:
 ref=proof(p);v=read(p);cfg=v['config'];assert proof(cfg['path'])==cfg;assert cfg['bytes']==2404 and cfg['sha256']=='fe8ec7c1754d5bbdcaa22c83154a258032ee5786c9e3e2b6c156b18853eef10b'
 assert v['status']=='FROZEN_FOUR_BANK_REAL_G10_124194_NATIVE_JOINT_INPUTS_ACCEPTED'and v['questions']==124194 and v['trajectories']==1241940 and v['source_frozen_batches']==971 and v['actor_updates']==v['probe_updates']==0 and v['final_outcomes_read']is False and v['complete169028_claim']is False
 assert v['groups']==v['outputs']['groups.json']and v['features_layer26']==v['outputs']['features-layer26.json']
 for r in v['outputs'].values():assert proof(r['path'])==r
 assert{r['key']for r in v['source_refs']}=={'existing58432','recovered1216','audited243','audited262'}
 out=dict(status='ACTUAL_FULL124194_ROOT_JOIN_EXACT_BINDING_GATE_PASSED',epoch=time.time(),complete=ref,config=cfg,groups=v['groups'],features=v['features_layer26'],questions=v['questions'],trajectories=v['trajectories'],source_frozen_batches=v['source_frozen_batches'],source_refs=v['source_refs'],source_metadata_only=True,real_tensor_or_checkpoint_loaded=False,GPU_submitted=False)
print(json.dumps(out))
'''
spec=importlib.util.spec_from_file_location('native_main_final_bank_only',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl);ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0']).decode().strip();packed=base64.b64encode(zlib.compress(BODY.encode())).decode();cmd='python3 -c '+shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(packed)+')))');r=ctl.run(cmd,account='mou176',source_address=ip,timeout=30)
if r.returncode:raise RuntimeError(r.stdout[-3000:]+r.stderr[-1000:])
v=json.loads(r.stdout.splitlines()[-1]);Path(__file__).with_name('actual-final-bank-gate.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v))
