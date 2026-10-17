#!/bin/sh
# Verify current publication; read the actually bound frozen runtime from main state.
python3 - <<'PY'
import subprocess,tempfile,pathlib,hashlib,json,datetime,shutil
repo=pathlib.Path.cwd();root=pathlib.Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control')
def run(args,cwd=repo,data=None):return subprocess.run(args,cwd=cwd,input=data,capture_output=True,check=True,timeout=180).stdout
head=run(['git','rev-parse','HEAD']).decode().strip();branch=run(['git','branch','--show-current']).decode().strip();url=run(['git','remote','get-url','origin']).decode().strip();remote=run(['git','ls-remote','origin','refs/heads/'+branch]).decode().split()[0];assert remote==head
bare=pathlib.Path(tempfile.mkdtemp(prefix='actorhidden-independent-git-'))
try:
 run(['git','init','--bare',str(bare)]);run(['git','fetch','--depth=1','--no-tags',url,'refs/heads/'+branch],cwd=bare);assert run(['git','rev-parse','FETCH_HEAD'],cwd=bare).decode().strip()==head;assert not (bare/'objects/info/alternates').exists();tree=run(['git','ls-tree','-r','-z',head]);assert run(['git','ls-tree','-r','-z',head],cwd=bare)==tree
 files=[]
 for line in tree.split(b'\0'):
  if line:
   meta,path=line.split(b'\t',1);mode,kind,oid=meta.decode().split()
   if kind=='blob':files.append((path.decode(),oid))
 ids=list(dict.fromkeys(oid for _,oid in files));request=('\n'.join(ids)+'\n').encode()
 def content(cwd):
  raw=run(['git','cat-file','--batch'],cwd=cwd,data=request);out={};offset=0
  for oid in ids:
   nl=raw.index(b'\n',offset);h=raw[offset:nl].decode().split();assert h[:2]==[oid,'blob'];n=int(h[2]);b=raw[nl+1:nl+1+n];assert len(b)==n;out[oid]=(n,hashlib.sha256(b).hexdigest());offset=nl+2+n
  assert offset==len(raw);return out
 local=content(repo);downloaded=content(bare);assert local==downloaded;rows=[dict(path=p,git_blob_oid=oid,bytes=local[oid][0],sha256=local[oid][1],independently_downloaded_full_sha256=downloaded[oid][1],match=True) for p,oid in files];proof=dict(status='PUSHED_ALL_GIT_BLOBS_INDEPENDENT_FULL_SHA256_READBACK_PASS',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),commit=head,remote_sha=remote,branch=branch,git_files=len(rows),unique_git_blobs=len(ids),fresh_bare_fetch=True,object_alternates_used=False,all_files_full_sha256_match=True,training_runtime_modified=False,training_runtime_commit=json.loads((root/'work/search-g1-actorhidden-period20-state.json').read_text())['current_runtime_commit'],manifest=rows);p=root/'work'/('actorhidden-period20-'+head[:8]+'-independent-readback.json');p.write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(dict(commit=head,files=len(rows),proof=str(p),proof_sha256=hashlib.sha256(p.read_bytes()).hexdigest())))
finally:shutil.rmtree(bare)
PY