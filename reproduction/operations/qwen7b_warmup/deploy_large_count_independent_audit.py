"""Publish/stage one small CPU auditor; never submit or alter GPU work."""
import argparse
import base64
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess

P='reproduction/operations/qwen7b_warmup/'
FILES=[P+n for n in ('direct_count_large_independent_audit.py','test_direct_count_large_independent_audit.py','deploy_large_count_independent_audit.py')]
BASE='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/direct-count-large-independent-audit-v2'
BRANCH='codex/large-count-independent-audit-20261021'
COMMON=r'''
import pathlib,json,hashlib,os,fcntl,time,subprocess,base64,gzip,resource
b=pathlib.Path(v['base']);b.mkdir(exist_ok=True);code=b/('code-'+v['commit'][:12]);code.mkdir(exist_ok=True)
def proof(p):
 p=pathlib.Path(p);raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def read(p):return json.loads(pathlib.Path(p).read_bytes())
def immutable(p,val):
 raw=json.dumps(val,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()+b'\n'
 if p.exists():assert p.read_bytes()==raw,'Own immutable audit receipt differs'
 else:p.write_bytes(raw)
lock=(b/'operator.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
'''
CHUNK=r'''
data=base64.b64decode(v['part']);assert hashlib.sha256(data).hexdigest()==v['part_sha'];p=code/('part-%04d'%v['index'])
if p.exists():assert p.read_bytes()==data
else:p.write_bytes(data)
print(json.dumps(dict(status='SMALL_CPU_AUDITOR_CODE_CHUNK_ACCEPTED',index=v['index'])))
'''
STAGE=r'''
data=b''.join((code/('part-%04d'%i)).read_bytes()for i in range(v['parts']));assert hashlib.sha256(data).hexdigest()==v['bundle_sha'];payload=json.loads(gzip.decompress(data));assert set(payload)==set(v['files'])
for name,text in payload.items():
 raw=base64.b64decode(text);item=v['files'][name];assert len(raw)==item['bytes']and hashlib.sha256(raw).hexdigest()==item['sha256'];p=code/name
 if p.exists():assert p.read_bytes()==raw
 else:p.write_bytes(raw);p.chmod(0o444)
packet=read(code/'input-packet.json');assert packet['role']=='INDEPENDENT_FROZEN_LARGE_COUNT_NODE_AUDIT'and not packet['external_pool_access']and not packet['final_pool_access']
for item in (packet['config'],packet['frozen_NQ_halves']):assert proof(item['path'])==item
assert packet['target'].endswith('/direct-count-frozenE1-fit30016-hold5120-v2'),'Do not use failed v1 as current'
pub=dict(commit=v['commit'],files={name:proof(code/name)for name in payload},code_root=str(code),GPU_operations=False,frozen_packet=proof(code/'input-packet.json'))
immutable(b/'source-publication.json',pub)
fixture="import importlib.util,sys,unittest; from pathlib import Path; root=Path("+repr(str(code))+"); name='reproduction.operations.qwen7b_warmup.direct_count_large_independent_audit';spec=importlib.util.spec_from_file_location(name,root/'direct_count_large_independent_audit.py');m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);spec=importlib.util.spec_from_file_location('reproduction.operations.qwen7b_warmup.test_direct_count_large_independent_audit',root/'test_direct_count_large_independent_audit.py');t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t);r=unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromModule(t));assert r.wasSuccessful()"
# The pure-stdlib test's relative import needs only empty namespace packages.
fixture="import types;\nfor x in ('reproduction','reproduction.operations','reproduction.operations.qwen7b_warmup'):\n p=types.ModuleType(x);p.__path__=[];__import__('sys').modules[x]=p\n"+fixture
r=subprocess.run(['python3','-c',fixture],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True,timeout=15);(b/'fixtures.log').write_text(r.stdout);assert r.returncode==0,r.stdout[-1800:]
print(json.dumps(dict(status='INDEPENDENT_CPU_AUDITOR_STAGED_FIVE_FIXTURES_PASSED',publication=proof(b/'source-publication.json'),fixtures=proof(b/'fixtures.log'))))
'''
RUN=r'''
pub=read(b/'source-publication.json');assert pub['commit']==v['commit']
for item in pub['files'].values():assert proof(item['path'])==item
def limits():
 os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2]);resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3))
started=time.time();r=subprocess.run(['python3',str(code/'direct_count_large_independent_audit.py'),'--input-packet',str(code/'input-packet.json'),'--output',str(b/'audits')],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=42,preexec_fn=limits)
if r.returncode:raise RuntimeError(r.stderr[-2200:])
value=json.loads(r.stdout);answer=dict(status='FINITE_AVAILABLE_NODE_AUDIT_RETURNED',result=value,seconds=time.time()-started,returncode=r.returncode,publication=proof(b/'source-publication.json'),GPU=0)
name='run-'+hashlib.sha256(r.stdout.encode()).hexdigest()[:16]+'.json';immutable(b/name,answer)
print(json.dumps(dict(answer,execution_receipt=proof(b/name))))
'''
STATUS=r'''
packet=read(code/'input-packet.json');target=pathlib.Path(packet['target']);out=dict(epoch=time.time(),source=proof(b/'source-publication.json')if(b/'source-publication.json').exists()else None,cache_complete=(target/'cache/complete.json').exists(),nodes=[])
for epoch in (0,1,2,4,8,20):
 p=target/'evaluations'/('epoch-%02d'%epoch)/'complete.json'
 if p.exists():out['nodes'].append(dict(epoch=epoch,receipt=proof(p),independent_raw=(b/'audits'/('epoch-%02d-raw.json'%epoch)).exists(),independent_EMA=(b/'audits'/('epoch-%02d-EMA.json'%epoch)).exists()))
print(json.dumps(out))
'''


def main():
 p=argparse.ArgumentParser();p.add_argument('--control-root',type=Path,required=True);p.add_argument('--commit',required=True);p.add_argument('--input-packet',type=Path,required=True);p.add_argument('--action',choices=('stage','run','status'),required=True);a=p.parse_args();repo=Path(__file__).resolve().parents[3];files={};payload={};packet_name=str(a.input_packet.resolve().relative_to(repo))
 for name in FILES+[packet_name]:
  raw=subprocess.check_output(['git','show',a.commit+':'+name],cwd=repo);assert raw==(repo/name).read_bytes();short='input-packet.json'if name==packet_name else Path(name).name;payload[short]=base64.b64encode(raw).decode();files[short]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
 values=dict(base=BASE,commit=a.commit,files=files);blocks=dict(chunk=CHUNK,stage=STAGE,run=RUN,status=STATUS)
 spec=importlib.util.spec_from_file_location('large_count_audit_private_control',a.control_root/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl);ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip()
 def call(action,extra=None):
  script='v='+repr(dict(values,**(extra or {})))+'\n'+COMMON+'\n'+blocks[action];compile(script,action,'exec');command='python3 -c '+shlex.quote(script);assert len(command.encode())<16000;r=ctl.run(command,account='mou176',source_address=ip,timeout=55);lines=[json.loads(x)for x in r.stdout.decode(errors='replace').splitlines()if x.startswith('{')]
  if r.returncode or not lines:raise RuntimeError(r.stdout.decode(errors='replace')[-2000:]+r.stderr.decode(errors='replace')[-1800:])
  return lines[-1]
 if a.action=='stage':
  remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+BRANCH],cwd=repo,text=True).split()[0];assert remote==a.commit;blob=gzip.compress(json.dumps(payload,sort_keys=True).encode(),mtime=0)
  for i,start in enumerate(range(0,len(blob),4500)):
   part=blob[start:start+4500];call('chunk',dict(index=i,part=base64.b64encode(part).decode(),part_sha=hashlib.sha256(part).hexdigest()))
  answer=call('stage',dict(parts=(len(blob)+4499)//4500,bundle_sha=hashlib.sha256(blob).hexdigest()))
 else:answer=call(a.action)
 out=a.control_root/'outputs/qwen7b-warmup-20261019/reports'/('large-count-independent-'+a.action+'.json');out.parent.mkdir(exist_ok=True,parents=True);out.write_text(json.dumps(answer,indent=2,sort_keys=True)+'\n');print(json.dumps(answer))


if __name__=='__main__':main()
