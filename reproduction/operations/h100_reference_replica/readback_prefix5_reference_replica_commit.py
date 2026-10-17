from pathlib import Path
import subprocess,json,hashlib,datetime,argparse
p=argparse.ArgumentParser();p.add_argument('--commit',required=True);a=p.parse_args();repo=Path('/Users/shaoshitong/Documents/Search-G1/h100-prefix5-reference-replica-20261018');bare=Path('work/actorhidden-ffb97281-fresh-readback.git').resolve();assert not (bare/'objects/info/alternates').exists();branch='codex/h100-prefix5-reference-replica-prepare-20261018';subprocess.run(['git','fetch','--depth=1','git@github.com:shaoshitong/Search-R1.git',branch],cwd=bare,check=True,capture_output=True);remote=subprocess.check_output(['git','rev-parse','FETCH_HEAD'],cwd=bare,text=True).strip();assert remote==a.commit,(remote,a.commit)
rows=[];objects=[]
for raw in subprocess.check_output(['git','ls-tree','-rz',a.commit],cwd=repo).split(b'\0'):
 if not raw:continue
 meta,name=raw.split(b'\t',1);mode,kind,oid=meta.split();assert kind==b'blob';rows.append((name.decode(),oid.decode()));objects.append(oid)
objects=list(dict.fromkeys(objects));payload=b'\n'.join(objects)+b'\n'
def blob_hashes(directory):
 r=subprocess.run(['git','cat-file','--batch'],cwd=directory,input=payload,capture_output=True,check=True);raw=r.stdout;cursor=0;result={}
 for oid in objects:
  stop=raw.index(b'\n',cursor);header=raw[cursor:stop].split();assert header[:2]==[oid,b'blob'];size=int(header[2]);cursor=stop+1;data=raw[cursor:cursor+size];assert len(data)==size and raw[cursor+size:cursor+size+1]==b'\n';cursor+=size+1;result[oid.decode()]=dict(bytes=size,sha256=hashlib.sha256(data).hexdigest())
 assert cursor==len(raw);return result
local=blob_hashes(repo);downloaded=blob_hashes(bare);assert local==downloaded
proof=dict(status='PUSHED_ALL_GIT_BLOBS_INDEPENDENT_FULL_SHA256_READBACK_PASS',commit=a.commit,remote_sha=remote,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),git_files=len(rows),unique_git_blobs=len(objects),all_files_full_sha256_match=True,independent_object_store=True,object_alternates_used=False,manifest=[dict(path=name,git_blob_oid=oid,**local[oid])for name,oid in rows]);out=Path('work')/('prefix5-reference-replica-'+a.commit[:8]+'-independent-readback.json');out.write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps({k:v for k,v in proof.items()if k!='manifest'}));print('proof',str(out.resolve()),'SHA',hashlib.sha256(out.read_bytes()).hexdigest())
