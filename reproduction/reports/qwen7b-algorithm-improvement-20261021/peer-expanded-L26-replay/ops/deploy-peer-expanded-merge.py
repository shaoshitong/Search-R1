"""Publish-backed independent CPU merger; never mutates active capture code."""
from pathlib import Path
import argparse,base64,hashlib,importlib.util,json,shlex,subprocess
C=Path(__file__).resolve().parents[2];R=C/'work/qwen7b-probe-warmup-20261019';P=C/'outputs/qwen7b-warmup-20261019/reports';PACKAGE='reproduction/operations/qwen7b_warmup/'
A='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/peer-expanded-L26-replay-v1'
ap=argparse.ArgumentParser();ap.add_argument('--commit',required=True);ap.add_argument('--action',choices=('stage','run','status'),required=True);a=ap.parse_args()
remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/peer-expanded-L26-merge-20261021'],cwd=R,text=True).split()[0];assert remote==a.commit
s=importlib.util.spec_from_file_location('private_cpu_merge',C/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(s);s.loader.exec_module(ctl);ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip()
common='''import pathlib,json,hashlib,os,subprocess,time,fcntl
b=pathlib.Path(BASE);d=b/('cpu-merge-code-'+COMMIT[:12]);d.mkdir(exist_ok=True);guard=(d/'operator.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
def read(p):return json.loads(pathlib.Path(p).read_text())
def fp(p):
 p=pathlib.Path(p);data=p.read_bytes();return dict(path=str(p),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
def atomic(p,v):
 p=pathlib.Path(p);q=p.with_name(p.name+'.%d.tmp'%os.getpid());q.write_text(json.dumps(v,sort_keys=True)+'\\n');q.replace(p)
def ticks(pid):
 try:return pathlib.Path('/proc/%d/stat'%pid).read_text().rsplit(')',1)[1].split()[19]
 except FileNotFoundError:return None
pub=read(b/'source-publication.json');root=pathlib.Path(pub['code_root']);assert pub['commit']=='bea03d5f1d908f357c368684b9e8c086ca0e687b' and root.parent==b
'''.replace('BASE',repr(A)).replace('COMMIT',repr(a.commit))
def call(code):
 command='python3 -c '+shlex.quote(common+'\n'+code);assert len(command.encode())<16000
 r=ctl.run(command,account='mou176',source_address=ip,timeout=55);rows=[json.loads(x) for x in r.stdout.decode(errors='replace').splitlines() if x.startswith('{')]
 if r.returncode or not rows:raise RuntimeError(r.stderr.decode(errors='replace')[-1000:]+' '+r.stdout.decode(errors='replace')[-1500:])
 return rows[-1]
if a.action=='stage':
 files={}
 for name in ('peer_expanded_l26_merge.py','test_peer_expanded_l26_merge.py'):
  data=subprocess.check_output(['git','show',a.commit+':'+PACKAGE+name],cwd=R);assert data==(R/(PACKAGE+name)).read_bytes();encoded=base64.b64encode(data).decode();sha=hashlib.sha256(data).hexdigest()
  value=call('''import base64
x=base64.b64decode(ENCODED);assert hashlib.sha256(x).hexdigest()==SHA;p=d/NAME
if p.exists():assert p.read_bytes()==x
else:p.write_bytes(x);p.chmod(0o444)
print(json.dumps(fp(p)))
'''.replace('ENCODED',repr(encoded)).replace('SHA',repr(sha)).replace('NAME',repr(name)));files[name]=value
 loader="import importlib.util,sys,pathlib; base=pathlib.Path("+repr(A+'/cpu-merge-code-'+a.commit[:12])+ "); p=base/'peer_expanded_l26_merge.py'; s=importlib.util.spec_from_file_location('reproduction.operations.qwen7b_warmup.peer_expanded_l26_merge',p); m=importlib.util.module_from_spec(s); sys.modules[s.name]=m; s.loader.exec_module(m); "
 tests=loader+"p=base/'test_peer_expanded_l26_merge.py'; s=importlib.util.spec_from_file_location('reproduction.operations.qwen7b_warmup.test_peer_expanded_l26_merge',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); import unittest; result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(m)); assert result.wasSuccessful()"
 value=call('''files=FILES
for item in files.values():assert fp(item['path'])==item
for name,item in pub['files'].items():assert fp(root/name)==dict(item,path=str(root/name))
env=dict(os.environ,QWEN7B_CODE_ROOT=str(root),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
r=subprocess.run(['bash',str(root/'reproduction/operations/qwen7b_warmup/runtime.sh'),'-c',TESTS],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True,timeout=40);(d/'cpu-tests.log').write_text(r.stdout);assert r.returncode==0,r.stdout[-1500:]
value=dict(status='IMMUTABLE_METADATA_ONLY_MERGE_SOURCE_AND_CPU_TESTS_ACCEPTED',commit=COMMIT,files=files,foundation=fp(b/'source-publication.json'),log=fp(d/'cpu-tests.log'),loader=LOADER);p=d/'publication.json'
if p.exists():assert read(p)==value
else:atomic(p,value)
print(json.dumps(dict(status=value['status'],publication=fp(p))))
'''.replace('FILES',repr(files)).replace('TESTS',repr(tests)).replace('COMMIT',repr(a.commit)).replace('LOADER',repr(loader)))
elif a.action=='run':
 value=call('''publication=read(d/'publication.json')
for item in publication['files'].values():assert fp(item['path'])==item
assert fp(b/'source-publication.json')==publication['foundation']
if (b/'bank/complete.json').exists():print(json.dumps(dict(status='EXISTING_MERGED_BANK_PRESERVED',complete=fp(b/'bank/complete.json'))))
elif not all((b/('shard-%02d'%i)/'capture/complete.json').exists() for i in range(4)):print(json.dumps(dict(status='WAITING_ALL_FOUR_ACTUAL_FEATURE_COMPLETIONS_NO_MERGE_STARTED')))
elif (d/'launch.json').exists():
 v=read(d/'launch.json');print(json.dumps(dict(status='EXISTING_CPU_MERGE_PROCESS_PRESERVED',launch=v,same_process_birth=ticks(v['pid'])==v['start_ticks'])))
else:
 assert not (d/'launch-intent.json').exists(),'Unknown prior launch; preserve and inspect'
 atomic(d/'launch-intent.json',dict(publication=fp(d/'publication.json'),time=time.time()))
 program=publication['loader']+"import json; print(json.dumps(m.merge("+repr(str(b))+"),sort_keys=True))"
 env=dict(os.environ,QWEN7B_CODE_ROOT=str(root),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
 with (d/'merge.log').open('ab') as log:child=subprocess.Popen(['bash',str(root/'reproduction/operations/qwen7b_warmup/runtime.sh'),'-c',program],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 v=dict(status='CPU_MERGE_STARTED_NOT_YET_ACCEPTED',pid=child.pid,start_ticks=ticks(child.pid),time=time.time());atomic(d/'launch.json',v);print(json.dumps(v))
''')
else:
 value=call('''v=dict(status='ACTUAL_CPU_MERGE_OBSERVATION',time=time.time())
if (d/'launch.json').exists():v['launch']=read(d/'launch.json');v['same_process_birth']=ticks(v['launch']['pid'])==v['launch']['start_ticks']
if (d/'merge.log').exists():v['log_tail']=(d/'merge.log').read_text()[-1500:]
if (b/'bank/complete.json').exists():v['complete']=read(b/'bank/complete.json');v['complete_ref']=fp(b/'bank/complete.json')
print(json.dumps(v))
''')
(P/('peer-expanded-L26-merge-'+a.action+'.json')).write_text(json.dumps(value,sort_keys=True,indent=2)+'\n');print(json.dumps(value))
