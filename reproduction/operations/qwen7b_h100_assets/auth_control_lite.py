"""Credential-only stdin control; no credential files, argv or model transit."""
import argparse, getpass, hashlib, importlib.util, json, shlex, subprocess, os
from pathlib import Path
ROOT=Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control')
REMOTE='/data/shaoshitong/qwen7b-h100-raw-transfer-20261020'
PYTHON='/home/shaoshitong/anaconda3/bin/python'

def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['preflight','bridge','receiver']);a=p.parse_args()
 token=getpass.getpass('HF token (RAM only): ')
 secret=dict(hf_token=token)
 if a.action=='preflight':
  code='''import os,json,sys
os.environ.update(HTTP_PROXY='http://127.0.0.1:17894',HTTPS_PROXY='http://127.0.0.1:17894',HF_HUB_DISABLE_XET='1',HF_HUB_DISABLE_TELEMETRY='1')
from huggingface_hub import HfApi
s=json.loads(sys.stdin.readline())
try:
 a=HfApi(token=s['hf_token']);w=a.whoami();assert w['name'].lower()=='sutoonq';repo='Sutoonq/Qwen25-7B-Probe-H100-Transfer-20261020';a.create_repo(repo,repo_type='dataset',private=True,exist_ok=True);v=a.repo_info(repo,repo_type='dataset');assert v.private
 print(json.dumps(dict(status='PRIVATE_HF_WRITE_PREFLIGHT_PASS',account=w['name'],repo=repo,private=True)))
except Exception as e:
 print(json.dumps(dict(status='HF_PREFLIGHT_FAILED',error_type=type(e).__name__,http_status=getattr(getattr(e,'response',None),'status_code',None))));sys.exit(1)
'''
 elif a.action=='bridge':
  spec=importlib.util.spec_from_file_location('hpc','/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);secret['hpc_password']=m.password_for('mou176').decode()
  raw=Path(__file__).with_name('hf_bridge_lite.py').read_bytes();digest=hashlib.sha256(raw).hexdigest()
  code='SHA='+repr(digest)+'\n'+'''import os,json,sys,subprocess,time,hashlib
from pathlib import Path
s=json.loads(sys.stdin.readline());b=Path('/data/shaoshitong/qwen7b-h100-raw-transfer-20261020');p=b/('hf-bridge-'+SHA+'.py');assert hashlib.sha256(p.read_bytes()).hexdigest()==SHA;r=b/'bridge-launch-receipt.json'
if r.exists():print(json.dumps(dict(status='EXISTING_BRIDGE_RECEIPT_NO_REPLAY',receipt=json.loads(r.read_text()))))
else:
 with(b/'bridge.log').open('xb')as f:q=subprocess.Popen(['/home/shaoshitong/anaconda3/bin/python',str(p)],stdin=subprocess.PIPE,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,HF_HUB_DISABLE_XET='1',HF_HUB_DISABLE_PROGRESS_BARS='1',HF_HUB_VERBOSITY='error'))
 q.stdin.write((json.dumps(s)+'\\n').encode());q.stdin.close();v=dict(pid=q.pid,start_ticks=Path('/proc/%d/stat'%q.pid).read_text().rsplit(')',1)[1].split()[19],source_sha256=SHA,epoch=time.time(),credentials_saved=False)
 with r.open('x')as f:json.dump(v,f)
 print(json.dumps(dict(status='BRIDGE_LAUNCHED_NOT_MIGRATION_COMPLETE',receipt=v)))
'''
 else:
  raw=Path(__file__).with_name('receiver_lite.py').read_bytes();digest=hashlib.sha256(raw).hexdigest()
  code='SHA='+repr(digest)+'\n'+"""import os,json,sys,subprocess,time,hashlib
from pathlib import Path
s=json.loads(sys.stdin.readline());b=Path('/mnt/data/shaoshitong/qwen7b-probe-assets-20261020');p=b/'control'/('receiver-'+SHA+'.py');assert hashlib.sha256(p.read_bytes()).hexdigest()==SHA;r=b/'control/data-receiver-lite-launch-receipt.json'
if r.exists():print(json.dumps(dict(status='EXISTING_RECEIVER_RECEIPT_NO_REPLAY',receipt=json.loads(r.read_text()))))
else:
 with(b/'control/data-receiver-lite.log').open('xb')as f:q=subprocess.Popen(['/usr/bin/python3',str(p)],stdin=subprocess.PIPE,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',HF_HUB_DISABLE_PROGRESS_BARS='1'))
 q.stdin.write((json.dumps(s)+'\\n').encode());q.stdin.close();v=dict(pid=q.pid,start_ticks=Path('/proc/%d/stat'%q.pid).read_text().rsplit(')',1)[1].split()[19],source_sha256=SHA,epoch=time.time(),credentials_saved=False)
 with r.open('x')as f:json.dump(v,f)
 print(json.dumps(dict(status='RECEIVER_LAUNCHED_NOT_MIGRATION_COMPLETE',receipt=v)))
"""
 if a.action=='receiver':
  spec=importlib.util.spec_from_file_location('ctl','/Users/shaoshitong/Documents/remote_codex_control/control.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  spec=importlib.util.spec_from_file_location('route','/Users/shaoshitong/Documents/Search-G1/h100-prefix5-reference-replica-20261018/reproduction/operations/h100_reference_replica/readonly_management_route.py');route=importlib.util.module_from_spec(spec);spec.loader.exec_module(route);os.environ['PREFIX5_AB_MGMT_EXISTING_PROXY']='1'
  argv=route.management_argv(m.route('h100-10',shlex.join(['/usr/bin/python3','-c',code])))
 else:argv=['ssh','-T','-oBatchMode=yes','-oConnectTimeout=8','hkust2',shlex.join([PYTHON,'-c',code])]
 r=subprocess.run(argv,input=json.dumps(secret)+'\n',text=True,capture_output=True,timeout=80)
 # Never save the stdin payload. The remote emits sanitized scalar metadata only.
 out=ROOT/('work/qwen7b-hf-lite-'+a.action+'-sanitized-result.json');out.write_text(json.dumps(dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr),indent=2)+'\n');print(r.stdout);print('returncode',r.returncode);raise SystemExit(r.returncode)

if __name__=='__main__':main()
