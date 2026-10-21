"""CPU-only stage/prepare/status; deliberately no sbatch or submit action."""
import argparse,base64,hashlib,importlib.util,json,pathlib,shlex,subprocess
REPO=pathlib.Path(__file__).resolve().parents[3];CONTROL=REPO.parents[1]
PKG='reproduction/operations/qwen7b_warmup/'
OWN=('direct_count_large_capacity.py','direct_count_capacity.py','test_direct_count_large_capacity.py')
BASE='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/direct-count-frozenE1-fit30016-hold5120-v1'
REFERENCE='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/direct-hidden-count30016-train-v2'
HERE=REPO/'reproduction/reports/qwen7b-algorithm-improvement-20261021/direct-count-large-capacity'
PACKET=HERE/'INPUT-PACKET.json'
COMMON='''import pathlib,json,hashlib,fcntl,os,time,base64,subprocess,shutil
b=pathlib.Path(v['base']);b.mkdir(exist_ok=True);guard=(b/'operator.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
def read(p):return json.loads(pathlib.Path(p).read_bytes())
def proof(p):
 p=pathlib.Path(p);raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
'''
def main():
 p=argparse.ArgumentParser();p.add_argument('--action',choices=('stage','prepare','status'),required=True);a=p.parse_args()
 spec=importlib.util.spec_from_file_location('large_head_capacity_CPU_prepare_private',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl)
 ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0']).decode().strip()
 def call(body,**values):
  values.update(base=BASE,reference=REFERENCE,pkg=PKG);cmd='python3 -c '+shlex.quote('v='+repr(values)+'\n'+COMMON+body)
  if len(cmd.encode())>=16000:raise ValueError('Control command must stay below16KB')
  r=ctl.run(cmd,account='mou176',source_address=ip,timeout=55)
  if r.returncode:raise RuntimeError('CPU preparation not accepted; inspect preserved state: '+r.stderr.decode()[-2500:]+r.stdout.decode()[-2000:])
  return json.loads(next(x for x in reversed(r.stdout.decode().splitlines())if x.startswith('{')))
 if a.action=='stage':
  commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO).decode().strip();remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/qwen7b-probe-warmup-20261019'],cwd=REPO).decode().split()[0]
  if commit!=remote:raise ValueError('Publish exact source before CPU preparation')
  blobs={}
  for name in OWN:
   raw=subprocess.check_output(['git','show',commit+':'+PKG+name],cwd=REPO)
   if raw!=(REPO/PKG/name).read_bytes():raise ValueError('Unpublished local source refused')
   blobs[name]=raw
  blobs['input-packet.json']=PACKET.read_bytes();manifests={}
  for name,raw in blobs.items():
   manifests[name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),parts=(len(raw)+4095)//4096)
   for i,start in enumerate(range(0,len(raw),4096)):
    chunk=raw[start:start+4096]
    call("root=b/('code-'+v['commit'][:12]);root.mkdir(exist_ok=True);p=root/(v['name']+'.part%04d'%v['i']);raw=base64.b64decode(v['data']);assert hashlib.sha256(raw).hexdigest()==v['sha']\nif p.exists():assert p.read_bytes()==raw\nelse:p.write_bytes(raw)\nprint(json.dumps(dict(name=v['name'],part=v['i'])))",commit=commit,name=name,i=i,data=base64.b64encode(chunk).decode(),sha=hashlib.sha256(chunk).hexdigest())
  result=call("""root=b/('code-'+v['commit'][:12]);old=read(pathlib.Path(v['reference'])/'source-publication.json');assert old['commit']=='501cf80478b77f68c3e13a96d4659e3da0971b2e';files=dict(old['files'])
for name,ref in old['files'].items():
 src=pathlib.Path(old['code_root'])/name;assert proof(src)['sha256']==ref['sha256'];dst=root/name;dst.parent.mkdir(exist_ok=True,parents=True)
 if dst.exists():assert dst.read_bytes()==src.read_bytes()
 else:shutil.copyfile(str(src),str(dst));dst.chmod(0o444)
for name,ref in v['new'].items():
 raw=b''.join((root/(name+'.part%04d'%i)).read_bytes()for i in range(ref['parts']));assert len(raw)==ref['bytes'] and hashlib.sha256(raw).hexdigest()==ref['sha256']
 dst=b/name if name=='input-packet.json' else root/v['pkg']/name
 if name!='input-packet.json':compile(raw,name,'exec')
 if dst.exists():assert dst.read_bytes()==raw
 else:dst.write_bytes(raw);dst.chmod(0o444)
 if name!='input-packet.json':files[v['pkg']+name]=dict(bytes=len(raw),sha256=ref['sha256'])
pub=dict(commit=v['commit'],remote_sha=v['commit'],code_root=str(root),files=files,original_runtime_commit=old['commit'],input_packet=proof(b/'input-packet.json'),no_GPU_submission_action=True);path=b/'source-publication.json'
if path.exists():assert read(path)==pub
else:path.write_text(json.dumps(pub,sort_keys=True)+'\\n')
print(json.dumps(dict(status='LARGE_HEAD_CAPACITY_CODE_AND_INPUT_PACKET_STAGED_CPU_ONLY',publication=proof(path),input_packet=pub['input_packet'])))""",commit=commit,new=manifests)
 elif a.action=='prepare':
  result=call("""if (b/'preparation.json').exists():print(json.dumps(read(b/'preparation.json')))
else:
 pub=read(b/'source-publication.json');root=pub['code_root'];env=dict(os.environ,QWEN7B_CODE_ROOT=root,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
 code="import unittest;from reproduction.operations.qwen7b_warmup import direct_count_large_capacity as m; s=unittest.defaultTestLoader.loadTestsFromName('reproduction.operations.qwen7b_warmup.test_direct_count_large_capacity');r=unittest.TextTestRunner(verbosity=2).run(s);assert r.wasSuccessful();m.prepare("+repr(str(b/'input-packet.json'))+");print('metadata-only complete; no pretrained tensors loaded')"
 with(b/'cpu-prepare.log').open('ab')as log:r=subprocess.run(['bash',root+'/reproduction/operations/qwen7b_warmup/runtime.sh','-u','-c',code],stdout=log,stderr=subprocess.STDOUT,env=env,timeout=48)
 assert r.returncode==0,(b/'cpu-prepare.log').read_text()[-3500:]
 print(json.dumps(read(b/'preparation.json')))""")
 else:
  result=call("out=dict(epoch=time.time(),GPU_submitted=False)\nfor name in ('source-publication.json','preparation.json'):\n if(b/name).exists():out[name[:-5]]=read(b/name)\nif(b/'config.json').exists():out['config']=read(b/'config.json')\nif(b/'cpu-prepare.log').exists():out['log_tail']=(b/'cpu-prepare.log').read_text()[-3000:]\nprint(json.dumps(out))")
 HERE.mkdir(exist_ok=True);(HERE/('actual-'+a.action+'.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
