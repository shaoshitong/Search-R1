"""Publish-isolated CPU qualification/preparation of a stopped native four-rank joint."""
import argparse,base64,gzip,hashlib,json,subprocess,time
from .control import REPO,OUT,remote
SCRIPT=r'''
import pathlib,json,base64,gzip,hashlib,os,subprocess,time
physical=pathlib.Path('/mnt/data/shaoshitong');source=physical/SOURCE;base=physical/BASE
cleanup=json.loads((source/'logs/cleanup-status.json').read_text());assert cleanup['cleaned']and cleanup['owned_births_all_gone']
j=json.loads((source/'latest-joint.json').read_text());assert j['outer']==OUTER and len(j['actor']['rank_states'])==4
assert not base.exists(),'Reconcile existing namespace/receipt; never repeat preparation'
base.mkdir();(base/'repo').mkdir();(base/'preparation').mkdir();(base/'logs').mkdir()
packet=json.loads(gzip.decompress(base64.b64decode(PACKET)))
for row in packet['manifest']:
 rel=pathlib.Path(row['path']);assert not rel.is_absolute()and'..'not in rel.parts
 target=base/'repo'/rel;target.parent.mkdir(parents=True,exist_ok=True)
 if row['path']in packet['changed']:target.write_bytes(base64.b64decode(packet['changed'][row['path']]))
 else:
  original=source/'repo'/rel;assert original.is_file()and not original.is_symlink()
  assert hashlib.sha256(original.read_bytes()).hexdigest()==row['sha256'],str(rel)
  os.link(original,target)
 assert hashlib.sha256(target.read_bytes()).hexdigest()==row['sha256'],str(rel)
(base/'preparation/source-full-sha.json').write_text(json.dumps(dict(commit=COMMIT,files=packet['manifest'],checked_unix=time.time())))
canonical='/mnt/data/sihe-share5/workspace/training_data_node_10/shaoshitong/'
s=canonical+SOURCE;b=canonical+BASE
common=['unshare','--mount','--propagation','private','bash',str(base/'repo/reproduction/operations/b_replica_retained18/private_shared_paths.sh'),'env','CUDA_VISIBLE_DEVICES=','SEARCH_G1_PHYSICAL_WORLD_SIZE=4','SEARCH_G1_B_PREFIX_TOKENS=80','OMP_NUM_THREADS=2','MKL_NUM_THREADS=2','PYTHONDONTWRITEBYTECODE=1','PYTHONPATH='+b+'/repo','/dataset/shaoshitong/Search-G1-common8/env/bin/python','-u']
code="import subprocess;common="+repr(common[6:])+";q="+repr(['-m','reproduction.operations.b_replica_retained18.qualify_shared_latest','--source',s,'--outer',str(OUTER),'--output',s+'/preparation/recovery%d-cpu/full-state-proof.json'%OUTER,'--runtime-repo',b+'/repo'])+";p="+repr(['-m','reproduction.operations.b_replica_retained18.prepare_shared_latest','--source',s,'--base',b,'--outer',str(OUTER),'--commit',COMMIT])+";subprocess.run(common+q,check=True);subprocess.run(common+p,check=True)"
# One child owns qualification then preparation; no CUDA visibility or GPU writer.
command=common+['-c',code]
with(base/'logs/native-preparation.log').open('xb')as log:proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
st=pathlib.Path('/proc',str(proc.pid),'stat').read_text().rsplit(')',1)[1].split()
receipt=dict(pid=proc.pid,start_ticks=int(st[19]),command=command,commit=COMMIT,created_unix=time.time(),outer=OUTER,world_size=4,training_started=False)
(base/'preparation/stage-receipt.json').write_text(json.dumps(receipt));print(json.dumps(receipt))
'''
def main():
 p=argparse.ArgumentParser()
 for key in ('source','base','source-commit'):p.add_argument('--'+key,required=True)
 p.add_argument('--outer',type=int,required=True);a=p.parse_args()
 for name in (a.source,a.base):assert name.startswith('search-g1-')and'/'not in name and'..'not in name
 assert 181<a.outer<200
 assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO)
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip();branch=subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()
 assert subprocess.check_output(['git','ls-remote','origin','refs/heads/'+branch],cwd=REPO,text=True).split()[0]==commit
 changed=set(subprocess.check_output(['git','diff','--name-only',a.source_commit,commit],cwd=REPO,text=True).splitlines())
 tracked=subprocess.check_output(['git','ls-files','-z'],cwd=REPO).decode().strip('\0').split('\0');packet=dict(manifest=[],changed={})
 for name in tracked:
  data=(REPO/name).read_bytes();packet['manifest'].append(dict(path=name,sha256=hashlib.sha256(data).hexdigest()))
  if name in changed:packet['changed'][name]=base64.b64encode(data).decode()
 header='\n'.join(k+'='+repr(v)for k,v in dict(SOURCE=a.source,BASE=a.base,OUTER=a.outer,COMMIT=commit,PACKET=base64.b64encode(gzip.compress(json.dumps(packet).encode())).decode()).items())
 v=json.loads(remote(header+'\n'+SCRIPT,OUT/('same-four-stage-'+str(time.time_ns())+'-transport.json'),target='h100-10',established_controller=True))
 (OUT/('CP%d_SAME_FOUR_CPU_STAGE.json'%a.outer)).write_text(json.dumps(v,indent=2));print(json.dumps(v),flush=True)
if __name__=='__main__':main()
