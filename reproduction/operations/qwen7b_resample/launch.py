"""Once native control; code manifest only crosses Mac, never tensor/data."""
import base64,gzip,hashlib,importlib.util,io,json,os,shlex,subprocess,tarfile
from pathlib import Path
from reproduction.operations.qwen7b_h100_assets.deploy_code_direct import WORKER
ROOT=Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control');REPO=Path(__file__).resolve().parents[3]
BASE='/mnt/shared/workspace/training_data_node_10/shaoshitong/qwen7b-resample-seed20261021-g5-20261020'
def main():
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip();raw=subprocess.check_output(['git','archive','--format=tar.gz',commit],cwd=REPO);files={}
 with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz')as t:
  for m in t:
   if m.isdir():continue
   assert m.isfile();data=t.extractfile(m).read();files[m.name]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
 baseline=subprocess.check_output(['git','archive','--format=tar.gz','44944c26745213d7c2761037639f59be006b8b62'],cwd=REPO);reference={}
 with tarfile.open(fileobj=io.BytesIO(baseline),mode='r:gz')as t:
  for m in t:
   if m.isfile():reference[m.name]=hashlib.sha256(t.extractfile(m).read()).hexdigest()
 delta={}
 with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz')as t:
  for m in t:
   if m.isfile()and reference.get(m.name)!=files[m.name]['sha256']:delta[m.name]=base64.b64encode(t.extractfile(m).read()).decode()
 old=json.loads((ROOT/'work/qwen7b-resample-runtime-and-assets.json').read_text())['retrieval'];old['new_source_commit']=commit
 worker=WORKER.replace('/mnt/data/shaoshitong/qwen7b-probe-assets-20261020/code',BASE+'/code');payload=gzip.compress(json.dumps(dict(delta=delta,plan=[dict(role='repo',commit=commit,files=files)],worker=worker,entry=Path(__file__).with_name('entry.py').read_text(),retrieval=old)).encode())
 boot='BASE='+repr(BASE)+'\n'+r'''
import base64,gzip,hashlib,json,os,subprocess,sys,time
from pathlib import Path
unit='sst-q7b-newseed-g5-20261020-r5';local=Path('/root/.local/share/search-g1/qwen7b-newseed-g5/r5');local.mkdir(parents=True,exist_ok=True)
p=local/'intent.json'
if p.exists():print(json.dumps({'existing_intent':json.loads(p.read_text()),'not_replayed':True}));raise SystemExit(0)
for old in ['sst-q7b-newseed-g5-20261020-r3','sst-q7b-newseed-g5-20261020-r4']:
 assert subprocess.check_output(['systemctl','show',old,'-p','MainPID','--value'],text=True).strip()=='0'
v=json.loads(gzip.decompress(sys.stdin.buffer.read()))
with p.open('x')as f:json.dump(dict(unit=unit,epoch=time.time(),commit=v['plan'][0]['commit'],resume_existing_sealed_batches=True),f)
(local/'payload.json').write_text(json.dumps(v))
bootstrap="""import json,os,runpy,subprocess
from pathlib import Path
BASE=Path("""+repr(BASE)+""")
local=Path("""+repr(str(local))+""")
canonical=Path('/mnt/data/sihe-share5/workspace/training_data_node_10')
target=Path('/mnt/shared/workspace/training_data_node_10')
target.mkdir(parents=True,exist_ok=True)
subprocess.run(['mount','--bind',str(canonical),str(target)],check=True)
assert (BASE/'config.json').is_file()
v=json.loads((local/'payload.json').read_text());control=BASE/'attempts/r5';(control/'code').mkdir(parents=True,exist_ok=True)
(control/'code/delta.json').write_text(json.dumps(v['delta']));(control/'code/direct-plan.json').write_text(json.dumps(v['plan']));(control/'entry.py').write_text(v['entry']);(control/'retrieval-config.json').write_text(json.dumps(v['retrieval']))
os.environ['Q7B_CONTROL']=str(control);os.environ['Q7B_ATTEMPT']='r5'
runpy.run_path(str(control/'entry.py'),run_name='__main__')
"""
(local/'bootstrap.py').write_text(bootstrap)
wrapper='/root/.local/share/search-g1/source8-fault-recovery/b2d62068695e156dcc90fc42e112ad24c676a1e7531fc26e2c0c6a01dbbfc1c1/retained-fault-paths.sh'
args=['systemd-run','--unit='+unit,'--property=KillMode=control-group','--property=TimeoutStopSec=60','--property=Restart=no','--property=MemoryMax=256G','--property=TasksMax=2048','--property=StandardOutput=append:'+str(local/'native.log'),'--property=StandardError=append:'+str(local/'native.log'),'unshare','--mount','--propagation','private','bash',wrapper,'h100-11','/usr/bin/python3',str(local/'bootstrap.py')]
r=subprocess.run(args,capture_output=True,text=True);receipt=dict(unit=unit,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,epoch=time.time());(local/'receipt.json').write_text(json.dumps(receipt));print(json.dumps(receipt));assert r.returncode==0
'''
 for n,p in [('ctl','/Users/shaoshitong/Documents/remote_codex_control/control.py'),('route','/Users/shaoshitong/Documents/Search-G1/h100-prefix5-reference-replica-20261018/reproduction/operations/h100_reference_replica/readonly_management_route.py')]:
  spec=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);globals()[n]=m
 os.environ['PREFIX5_AB_MGMT_EXISTING_PROXY']='1';r=subprocess.run(route.management_argv(ctl.route('h100-11',shlex.join(['/usr/bin/python3','-c',boot]))),input=payload,capture_output=True,timeout=100);out=dict(returncode=r.returncode,stdout=r.stdout.decode(),stderr=r.stderr.decode());(ROOT/'work/qwen7b-resample-launch.json').write_text(json.dumps(out,indent=2));print(json.dumps(out));assert r.returncode==0
if __name__=='__main__':main()
