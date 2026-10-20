"""Exact four source snapshots, code-only Mac SSH payload, target full-file SHA."""
import hashlib, importlib.util, io, json, os, struct, subprocess, tarfile
from pathlib import Path
REPO=Path(__file__).resolve().parents[3]
ROOT=Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control')
REMOTE=r'''
import hashlib,io,json,os,struct,sys,tarfile,time
from pathlib import Path
b=Path('/mnt/data/shaoshitong/qwen7b-probe-assets-20261020/code');b.mkdir(exist_ok=True)
n=struct.unpack('!Q',sys.stdin.buffer.read(8))[0];assert n<5*1024**2;header=json.loads(sys.stdin.buffer.read(n));results=[]
for item in header:
 archive=b/('source-'+item['commit']+'.tar.gz');h=hashlib.sha256();left=item['archive_bytes']
 with archive.open('wb')as f:
  while left:
   raw=sys.stdin.buffer.read(min(left,4*1024**2));assert raw;f.write(raw);h.update(raw);left-=len(raw)
 assert h.hexdigest()==item['archive_sha256']
 dest=b/(item['role']+'-'+item['commit']);dest.mkdir(exist_ok=True)
 with tarfile.open(archive,'r:gz')as t:
  for m in t:
   assert not m.name.startswith('/')and'..'not in Path(m.name).parts
   p=dest/m.name;assert p.resolve().is_relative_to(dest.resolve())
   if m.isdir():p.mkdir(exist_ok=True,parents=True);continue
   assert m.isfile()and m.name in item['files'];p.parent.mkdir(parents=True,exist_ok=True)
   with t.extractfile(m)as f:raw=f.read()
   assert hashlib.sha256(raw).hexdigest()==item['files'][m.name]['sha256'];p.write_bytes(raw);p.chmod(m.mode&0o777)
 for name,row in item['files'].items():
  p=dest/name;assert p.stat().st_size==row['bytes']and hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
 manifest=b/('manifest-'+item['commit']+'.json');manifest.write_text(json.dumps(item,indent=2))
 results.append(dict(role=item['role'],commit=item['commit'],code_root=str(dest),files=len(item['files']),archive_sha256=item['archive_sha256'],all_target_files_fresh_sha_match=True,manifest=str(manifest)))
v=dict(status='ALL_SELECTED_CODE_SNAPSHOTS_FRESH_FULL_SHA_VERIFIED',snapshots=results,code_only_mac_transport=True,model_or_data_mac_transport=False,training_not_started=True,completed=time.time());p=b/'complete.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2));tmp.replace(p);print(json.dumps(v))
'''

def main():
 snapshots=[('sampling','f2d5ae7de92d198692abf5bf82143e309d91c271'),('warmup','96df5a23f23f6da0f24e79e67dcf0e3039f22de0'),('mixed_finetune','0eeb747d14ba11a1fc5ed1e3e4dec16f9073c435'),('migration','HEAD')]
 header=[];archives=[]
 for role,ref in snapshots:
  commit=subprocess.check_output(['git','rev-parse',ref],cwd=REPO,text=True).strip();raw=subprocess.check_output(['git','archive','--format=tar.gz',commit],cwd=REPO);files={}
  with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz')as t:
   for m in t:
    if m.isdir():continue
    assert m.isfile(),m.name;data=t.extractfile(m).read();files[m.name]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
  header.append(dict(role=role,commit=commit,archive_bytes=len(raw),archive_sha256=hashlib.sha256(raw).hexdigest(),files=files));archives.append(raw)
 h=json.dumps(header).encode();payload=struct.pack('!Q',len(h))+h+b''.join(archives)
 spec=importlib.util.spec_from_file_location('ctl','/Users/shaoshitong/Documents/remote_codex_control/control.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 spec=importlib.util.spec_from_file_location('route','/Users/shaoshitong/Documents/Search-G1/h100-prefix5-reference-replica-20261018/reproduction/operations/h100_reference_replica/readonly_management_route.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r);os.environ['PREFIX5_AB_MGMT_EXISTING_PROXY']='1'
 import shlex
 result=subprocess.run(r.management_argv(m.route('h100-10',shlex.join(['/usr/bin/python3','-c',REMOTE]))),input=payload,capture_output=True,timeout=120)
 p=ROOT/'work/qwen7b-h100-code-deployment.json';p.write_text(json.dumps(dict(returncode=result.returncode,stdout=result.stdout.decode(),stderr=result.stderr.decode()),indent=2)+'\n');assert result.returncode==0,result.stderr.decode()[-1500:];print(result.stdout.decode())

if __name__=='__main__':main()
