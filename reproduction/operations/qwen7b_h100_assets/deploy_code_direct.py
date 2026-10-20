"""Send only compressed code manifests; H100 fetches public pinned Git archives."""
import gzip,hashlib,importlib.util,io,json,os,shlex,subprocess,tarfile
from pathlib import Path
REPO=Path(__file__).resolve().parents[3]
ROOT=Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control')
WORKER=r'''
import hashlib,json,os,tarfile,time,requests
from pathlib import Path
b=Path('/mnt/data/shaoshitong/qwen7b-probe-assets-20261020/code')
def save(p,v):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for x in iter(lambda:f.read(1024**2),b''):h.update(x)
 return h.hexdigest()
try:
 plan=json.loads((b/'direct-plan.json').read_text());results=[]
 for item in plan:
  dest=b/(item['role']+'-'+item['commit']);dest.mkdir(exist_ok=True)
  archive=b/('github-'+item['commit']+'.tar.gz')
  if not archive.exists():
   for attempt in range(8):
    try:
     r=requests.get('https://codeload.github.com/shaoshitong/Search-R1/tar.gz/'+item['commit'],stream=True,timeout=(15,90));r.raise_for_status()
     partial=archive.with_suffix('.part')
     with partial.open('wb')as f:
      for x in r.iter_content(1024**2):f.write(x)
     partial.replace(archive);break
    except requests.RequestException:
     if attempt==7:raise
     time.sleep(3)
  seen=set()
  with tarfile.open(archive,'r:gz')as t:
   for m in t:
    parts=Path(m.name).parts;name='/'.join(parts[1:])
    if not name or m.isdir():continue
    assert m.isfile()and name in item['files'],name
    p=dest/name;assert not Path(name).is_absolute()and'..'not in Path(name).parts;p.parent.mkdir(parents=True,exist_ok=True)
    data=t.extractfile(m).read();assert len(data)==item['files'][name]['bytes']and hashlib.sha256(data).hexdigest()==item['files'][name]['sha256'],name
    p.write_bytes(data);p.chmod(m.mode&0o777);seen.add(name)
  assert seen==set(item['files'])
  for name,row in item['files'].items():
   p=dest/name;assert p.stat().st_size==row['bytes']and sha(p)==row['sha256']
  save(b/('manifest-'+item['commit']+'.json'),item)
  results.append(dict(role=item['role'],commit=item['commit'],code_root=str(dest),files=len(seen),all_target_files_fresh_sha_match=True))
  save(b/'direct-progress.json',dict(snapshots=results,epoch=time.time()))
 save(b/'complete.json',dict(status='ALL_SELECTED_CODE_SNAPSHOTS_FRESH_FULL_SHA_VERIFIED',snapshots=results,code_download='GITHUB_TO_H100',model_or_data_mac_transport=False,training_not_started=True,completed=time.time()))
except BaseException as e:save(b/'direct-failure.json',dict(error_type=type(e).__name__,detail=str(e)[:300],epoch=time.time()));raise
'''
def main():
 plan=[]
 for role,ref in [('sampling','f2d5ae7de92d198692abf5bf82143e309d91c271'),('warmup','96df5a23f23f6da0f24e79e67dcf0e3039f22de0'),('mixed_finetune','0eeb747d14ba11a1fc5ed1e3e4dec16f9073c435'),('migration','HEAD')]:
  commit=subprocess.check_output(['git','rev-parse',ref],cwd=REPO,text=True).strip();raw=subprocess.check_output(['git','archive','--format=tar.gz',commit],cwd=REPO);files={}
  with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz')as t:
   for m in t:
    if m.isdir():continue
    assert m.isfile();data=t.extractfile(m).read();files[m.name]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
  plan.append(dict(role=role,commit=commit,files=files))
 payload=gzip.compress(json.dumps(dict(plan=plan,worker=WORKER)).encode())
 boot=r'''
import gzip,hashlib,json,os,subprocess,sys,time
from pathlib import Path
b=Path('/mnt/data/shaoshitong/qwen7b-probe-assets-20261020/code');b.mkdir(exist_ok=True)
p=b/'direct-launch.json'
if p.exists():print(p.read_text());raise SystemExit(0)
v=json.loads(gzip.decompress(sys.stdin.buffer.read()));(b/'direct-plan.json').write_text(json.dumps(v['plan']))
s=b/('direct-worker-'+hashlib.sha256(v['worker'].encode()).hexdigest()+'.py');s.write_text(v['worker'])
f=(b/'direct-worker.log').open('ab');c=subprocess.Popen(['/usr/bin/python3',str(s)],stdin=subprocess.DEVNULL,stdout=f,stderr=f,start_new_session=True,env={**os.environ,'HTTP_PROXY':'http://10.33.0.18:3128','HTTPS_PROXY':'http://10.33.0.18:3128'})
r=dict(pid=c.pid,start_ticks=Path('/proc/%s/stat'%c.pid).read_text().rsplit(')',1)[1].split()[19],worker_sha256=hashlib.sha256(v['worker'].encode()).hexdigest(),epoch=time.time());p.write_text(json.dumps(r));print(json.dumps(r))
'''
 spec=importlib.util.spec_from_file_location('ctl','/Users/shaoshitong/Documents/remote_codex_control/control.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 spec=importlib.util.spec_from_file_location('route','/Users/shaoshitong/Documents/Search-G1/h100-prefix5-reference-replica-20261018/reproduction/operations/h100_reference_replica/readonly_management_route.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r);os.environ['PREFIX5_AB_MGMT_EXISTING_PROXY']='1'
 result=subprocess.run(r.management_argv(m.route('h100-10',shlex.join(['/usr/bin/python3','-c',boot]))),input=payload,capture_output=True,timeout=100)
 out=dict(returncode=result.returncode,stdout=result.stdout.decode(),stderr=result.stderr.decode(),compressed_control_bytes=len(payload));(ROOT/'work/qwen7b-code-direct-launch.json').write_text(json.dumps(out,indent=2));assert result.returncode==0;print(json.dumps(out))
if __name__=='__main__':main()
