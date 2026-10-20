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
b=Path(BASE);b.mkdir(exist_ok=True);(b/'code').mkdir(exist_ok=True);(b/'logs').mkdir(exist_ok=True);v=json.loads(gzip.decompress(sys.stdin.buffer.read()));p=b/'launch-intent.json';unit='sst-q7b-newseed-g5-20261020-r2'
if p.exists():print(json.dumps({'existing_intent':json.loads(p.read_text()),'unit':unit,'not_replayed':True}));raise SystemExit(0)
with p.open('x')as f:json.dump({'unit':unit,'epoch':time.time(),'source_commit':v['plan'][0]['commit'],'seed':20261021,'questions':169028,'trajectories':845140,'cards':[1,2,3]},f)
(b/'code/delta.json').write_text(json.dumps(v['delta']));(b/'code/direct-plan.json').write_text(json.dumps(v['plan']));(b/'code/fetch-code.py').write_text(v['worker']);(b/'entry.py').write_text(v['entry']);(b/'retrieval-config.json').write_text(json.dumps(v['retrieval']))
args=['systemd-run','--unit='+unit,'--setenv=HTTP_PROXY=http://10.33.0.18:3128','--setenv=HTTPS_PROXY=http://10.33.0.18:3128','--property=KillMode=control-group','--property=TimeoutStopSec=60','--property=Restart=no','--property=MemoryMax=256G','--property=TasksMax=2048','--property=StandardOutput=append:'+str(b/'logs/native.log'),'--property=StandardError=append:'+str(b/'logs/native.log'),'/usr/bin/python3',str(b/'entry.py')]
r=subprocess.run(args,env={**os.environ,'HTTP_PROXY':'http://10.33.0.18:3128','HTTPS_PROXY':'http://10.33.0.18:3128'},capture_output=True,text=True);receipt={'unit':unit,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'epoch':time.time()};(b/'launch-receipt.json').write_text(json.dumps(receipt));print(json.dumps(receipt));assert r.returncode==0
'''
 for n,p in [('ctl','/Users/shaoshitong/Documents/remote_codex_control/control.py'),('route','/Users/shaoshitong/Documents/Search-G1/h100-prefix5-reference-replica-20261018/reproduction/operations/h100_reference_replica/readonly_management_route.py')]:
  spec=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);globals()[n]=m
 os.environ['PREFIX5_AB_MGMT_EXISTING_PROXY']='1';r=subprocess.run(route.management_argv(ctl.route('h100-11',shlex.join(['/usr/bin/python3','-c',boot]))),input=payload,capture_output=True,timeout=100);out=dict(returncode=r.returncode,stdout=r.stdout.decode(),stderr=r.stderr.decode());(ROOT/'work/qwen7b-resample-launch.json').write_text(json.dumps(out,indent=2));print(json.dumps(out));assert r.returncode==0
if __name__=='__main__':main()
