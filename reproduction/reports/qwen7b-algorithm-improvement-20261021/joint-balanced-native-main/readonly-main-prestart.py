"""Single bounded metadata/queue check; no GPU, configuration write or submit."""
import base64,hashlib,importlib.util,json,shlex,subprocess,time,zlib
from pathlib import Path
CONTROL=Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5')
body='''import pathlib,json,hashlib,subprocess,fcntl,time
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/joint-balanced-native-E4-124194-v1')
def read(p):return json.loads(p.read_bytes())
def proof(p):
 raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
pub=read(b/'source-publication.json');assert pub['commit']=='3b56e583efe2469ac9108873c31de7144a5201e6';root=pathlib.Path(pub['code_root'])
for name,desc in pub['files'].items():
 actual=proof(root/name);assert actual['bytes']==desc['bytes']and actual['sha256']==desc['sha256'],'Immutable source differs'
lock=(b/'training.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);lock.close()
queries={}
for key,args in [('own_queue',['squeue','-h','-u','mou176','-o','%i|%j|%T|%R|%b']),('partition_states',['sinfo','-h','-p','acd_u','-o','%P|%a|%l|%D|%t|%G'])]:
 p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=15);queries[key]=dict(returncode=p.returncode,rows=p.stdout.strip().splitlines(),stderr=p.stderr[-500:]);assert p.returncode==0
bank=b.parent/'joint124194-native-training-bank-v1/complete.json'
print(json.dumps(dict(status='READONLY_NATIVE_MAIN_SOURCE_LOCK_QUEUE_SNAPSHOT',epoch=time.time(),publication=proof(b/'source-publication.json'),source_commit=pub['commit'],source_files_verified=len(pub['files']),config_exists=(b/'config.json').exists(),intent_exists=(b/'submission-intent.json').exists(),main_complete_exists=(b/'complete.json').exists(),actual_full124194_bank_complete_exists=bank.exists(),training_lock_free=True,queries=queries,GPU_submitted_by_this_check=False,configuration_changed=False)))
'''
spec=importlib.util.spec_from_file_location('readonly_native_main_prestart',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl);ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0']).decode().strip();packed=base64.b64encode(zlib.compress(body.encode())).decode();command='python3 -c '+shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(packed)+')))');r=ctl.run(command,account='mou176',source_address=ip,timeout=50)
if r.returncode:raise RuntimeError(r.stdout[-3000:]+r.stderr[-1000:])
v=json.loads(r.stdout.splitlines()[-1]);Path(__file__).with_name('actual-readonly-prestart.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v))
