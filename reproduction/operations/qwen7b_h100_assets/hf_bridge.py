"""HPC3 -> HKUST3 -> private HF. Secrets arrive only via stdin/RAM."""
import concurrent.futures, fcntl, hashlib, json, os, shlex, subprocess, sys, threading, time
from pathlib import Path

ROOT=Path('/backup/shaoshitong/qwen7b-h100-transfer-20261020')
SOURCE='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/migration-export-h100-20261020'
REPO='Sutoonq/Qwen25-7B-Probe-H100-Transfer-20261020'
SSH=['sshpass','-e','ssh','-T','-oStrictHostKeyChecking=yes','-oHostKeyAlias=hpc3login.hpc.hkust-gz.edu.cn','-oUserKnownHostsFile=/backup/shaoshitong/topd-hf-migration-20261009/hpc3-known-hosts','-oBatchMode=no','-oNumberOfPasswordPrompts=1','-oConnectTimeout=10','-oServerAliveInterval=20','-oServerAliveCountMax=3','-oControlMaster=no','-S','none','mou176@10.120.81.30']

def atomic(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.%d-%d.tmp'%(os.getpid(),threading.get_ident()));t.write_text(json.dumps(v,indent=2));t.replace(p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def main(secret):
 from huggingface_hub import HfApi
 ROOT.mkdir(parents=True,exist_ok=True);lock=(ROOT/'bridge.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 os.environ.update(HTTP_PROXY='http://127.0.0.1:17896',HTTPS_PROXY='http://127.0.0.1:17896',HF_HUB_DISABLE_XET='1',HF_HUB_DISABLE_PROGRESS_BARS='1',HF_HUB_DISABLE_TELEMETRY='1')
 api=HfApi(token=secret['hf_token']);who=api.whoami();assert who['name'].lower()=='sutoonq';api.create_repo(REPO,repo_type='dataset',private=True,exist_ok=True);assert api.repo_info(REPO,repo_type='dataset').private
 env=dict(os.environ,SSHPASS=secret['hpc_password']);secret['hpc_password']=None
 atomic(ROOT/'controller.json',dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],repo_id=REPO,private=True,account=who['name'],source=SOURCE,credentials_saved=False,mac_data_transit=False,started=time.time()))
 def remote(code,timeout=60):
  r=subprocess.run(SSH+['python3 -c '+shlex.quote(code)],env=env,capture_output=True,text=True,timeout=timeout)
  if r.returncode:raise RuntimeError('SOURCE_READ_RETURN_%d'%r.returncode)
  return json.loads(r.stdout)
 def catalog():
  return remote('BASE='+repr(SOURCE)+'\n'+'''import json
from pathlib import Path
b=Path(BASE);p=b/'plan.json';v=dict(plan_ready=p.is_file(),failure=json.loads((b/'failure.json').read_text())if(b/'failure.json').exists()else None)
if p.exists():
 s=json.loads(p.read_text());v.update(groups=[x['group']for x in s['groups']],files=s['files'],payload_bytes=s['payload_bytes'],revision=s['revision'],ready=[])
 for name in v['groups']:
  p=b/(name+'.json')
  if p.is_file():
   r=json.loads(p.read_text());v['ready'].append({k:r[k]for k in('group','archive','archive_bytes','archive_sha256','source_bytes')})
print(json.dumps(v))''')
 def upload(g):
  name=g['group'];done=ROOT/'uploaded'/(name+'.json')
  if done.exists():return json.loads(done.read_text())
  local=ROOT/'incoming'/(name+'.tar.zst');local.parent.mkdir(exist_ok=True)
  atomic(ROOT/'progress'/(name+'.json'),dict(stage='COPYING_HPC3_TO_HKUST3',expected_bytes=g['archive_bytes'],epoch=time.time()))
  if not local.exists()or local.stat().st_size!=g['archive_bytes']:
   args=['rsync','--partial','--append-verify','--protect-args','-e',shlex.join(SSH[:-1]),SSH[-1]+':'+g['archive'],str(local)]
   r=subprocess.run(args,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,timeout=3600)
   if r.returncode:raise RuntimeError('RSYNC_RETURN_%d'%r.returncode)
  assert local.stat().st_size==g['archive_bytes']and sha(local)==g['archive_sha256']
  receipt=remote('import json;print(json.dumps(json.load(open('+repr(SOURCE+'/'+name+'.json')+'))))')
  assert receipt['archive_sha256']==g['archive_sha256']and receipt['archive_bytes']==g['archive_bytes']
  atomic(ROOT/'progress'/(name+'.json'),dict(stage='UPLOADING_PRIVATE_HF_NOT_ACCEPTED',archive_bytes=g['archive_bytes'],epoch=time.time()))
  commit=api.upload_file(path_or_fileobj=str(local),path_in_repo='archives/'+local.name,repo_id=REPO,repo_type='dataset',commit_message='Verified source archive '+name)
  infos=api.get_paths_info(REPO,paths=['archives/'+local.name],repo_type='dataset');assert len(infos)==1 and infos[0].size==g['archive_bytes'];lfs=infos[0].lfs;oid=lfs.get('sha256')if isinstance(lfs,dict)else lfs.sha256;assert oid==g['archive_sha256']
  receipt.update(hf_repo=REPO,repo_type='dataset',private=True,hf_archive_path='archives/'+local.name,hf_archive_commit=commit.oid,hf_lfs_sha256=oid,intermediate_full_sha_verified=True)
  raw=json.dumps(receipt,sort_keys=True).encode();manifest_commit=api.upload_file(path_or_fileobj=raw,path_in_repo='receipts/'+name+'.json',repo_id=REPO,repo_type='dataset',commit_message='Source payload manifest '+name)
  accepted=dict(status='SOURCE_HKUST3_HF_ARCHIVE_SHA_PASS',group=name,archive_bytes=g['archive_bytes'],source_bytes=g['source_bytes'],archive_sha256=g['archive_sha256'],archive_commit=commit.oid,manifest_commit=manifest_commit.oid,hf_repo=REPO,private=True,completed=time.time());atomic(done,accepted);atomic(ROOT/'progress'/(name+'.json'),accepted);return accepted
 submitted=set();running={};summary_published=False
 with concurrent.futures.ThreadPoolExecutor(max_workers=2)as pool:
  while True:
   c=catalog()
   if c.get('failure'):atomic(ROOT/'source-failure-observed.json',c['failure'])
   if c['plan_ready']:
    if not summary_published:
     raw=json.dumps({k:c[k]for k in('groups','files','payload_bytes','revision')},sort_keys=True).encode();api.upload_file(path_or_fileobj=raw,path_in_repo='PLAN-SUMMARY.json',repo_id=REPO,repo_type='dataset',commit_message='Frozen selected migration scope');summary_published=True
    for g in c['ready']:
     if g['group']not in submitted:submitted.add(g['group']);running[pool.submit(upload,g)]=g
   for f in list(running):
    if f.done():
     g=running.pop(f)
     try:f.result()
     except Exception as e:
      n=ROOT/'progress'/(g['group']+'-retry.json');old=json.loads(n.read_text())if n.exists()else{};attempt=old.get('attempt',0)+1;atomic(n,dict(attempt=attempt,error_type=type(e).__name__,epoch=time.time()))
      if attempt>=8:atomic(ROOT/'failure.json',dict(stage='BRIDGE',group=g['group'],error_type=type(e).__name__,epoch=time.time()));raise
      submitted.remove(g['group'])
   uploaded=list((ROOT/'uploaded').glob('*.json'))if(ROOT/'uploaded').exists()else[]
   atomic(ROOT/'progress.json',dict(stage='SOURCE_TO_PRIVATE_HF',groups_verified=len(uploaded),groups_expected=len(c.get('groups',[])),hf_verified_archive_bytes=sum(json.loads(p.read_text())['archive_bytes']for p in uploaded),active_groups=[g['group']for g in running.values()],partial_archives=[dict(group=p.stem,bytes=p.stat().st_size)for p in(ROOT/'incoming').glob('*.tar.zst')],epoch=time.time()))
   if c['plan_ready']and len(uploaded)==len(c['groups'])and not running:break
   time.sleep(10)
 atomic(ROOT/'complete.json',dict(status='ALL_SELECTED_ARCHIVES_PRIVATE_HF_SHA_ACCEPTED_NOT_H100_COMPLETE',groups=len(c['groups']),files=c['files'],payload_bytes=c['payload_bytes'],repo_id=REPO,completed=time.time(),credentials_saved=False))

if __name__=='__main__':
 # Start only through an encrypted stdin bootstrap, not token/password arguments.
 secret=json.loads(sys.stdin.readline())
 try:main(secret)
 except BaseException as e:
  atomic(ROOT/'failure.json',dict(stage='BRIDGE_CONTROLLER',error_type=type(e).__name__,epoch=time.time()));raise SystemExit(1)
