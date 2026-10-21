"""Small terminal metadata audit; no checkpoint tensors on a login node."""
import base64,hashlib,importlib.util,json,shlex,subprocess,zlib
from pathlib import Path
CONTROL=Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5')
body='''import pathlib,json,hashlib,fcntl,subprocess,time
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/joint-balanced-native-E4-qualification-v3')
def proof(p):
 raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
values={n:json.loads((b/(n+'.json')).read_bytes())for n in('complete','first-update-replay','first-cp-independent-audit','durable-next-update-replay','progress','latest','config','source-publication','submission-result')}
assert values['complete']['cursor']==values['progress']['cursor']==values['latest']['cursor']==7
assert values['complete']['steps']==14425 and values['first-update-replay']['status']=='PASSED' and values['durable-next-update-replay']['status']=='PASSED'
assert values['first-cp-independent-audit']['status']=='NATIVE_JOINT_CP1_INDEPENDENT_CPU_FULL_SIX_STATE_AND_SAMPLER_RELOAD_PASSED'
assert values['first-cp-independent-audit']['allocation_job_id']=='745177'
assert values['first-update-replay']['block_changed']and values['first-update-replay']['readout_changed']
assert not(b/'failure.json').exists()
job=subprocess.check_output(['sacct','-n','-P','-j','745177','--format=JobIDRaw,State,ExitCode,Elapsed'],universal_newlines=True).strip().splitlines()
main=[r for r in job if r.split('|')[0]=='745177'];assert len(main)==1 and main[0].split('|')[1:3]==['COMPLETED','0:0']
f=(b/'training.lock').open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);f.close()
cp=values['latest'];assert pathlib.Path(cp['path']).stat().st_size==cp['bytes'];assert cp['sha256']=='c93c5eafb8393efc80bc615b91adf87cd24591d9dd11c3504a6282d36a1082cc'
out=dict(status='NATIVE_JOINT_REAL_7_UPDATE_QUALIFICATION_TERMINAL_ACCEPTED',epoch=time.time(),main_job=main[0],training_lock_free=True,terminal_receipts={n:proof(b/(n+'.json'))for n in values},complete=values['complete'],first_update=values['first-update-replay'],independent_CP1=values['first-cp-independent-audit'],durable_next_update=values['durable-next-update-replay'],source_commit=values['source-publication']['commit'],actual_config=values['config'],source_publication=values['source-publication'],source_checkpoint=values['config']['source_checkpoint'],no_login_checkpoint_tensor_load=True,CP7_full_SHA_inherited_from_producer_CPU_reload=True,external_evaluation_performed=False,full124194_main_training_claim=False)
print(json.dumps(out))
'''
spec=importlib.util.spec_from_file_location('native_qual_terminal',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl)
ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0']).decode().strip();packed=base64.b64encode(zlib.compress(body.encode())).decode();command='python3 -c '+shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(packed)+')))');r=ctl.run(command,account='mou176',source_address=ip,timeout=30)
if r.returncode:raise RuntimeError(r.stdout[-3000:]+r.stderr[-1000:])
v=json.loads(r.stdout.splitlines()[-1]);Path(__file__).with_name('actual-terminal-qualification.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps({k:v[k]for k in('status','main_job','source_commit','training_lock_free','complete')}))
