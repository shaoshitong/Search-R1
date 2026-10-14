import pathlib,json,hashlib,subprocess,sys,shlex,gzip
repo=pathlib.Path('/Users/shaoshitong/Documents/Search-G1/mixed-first-200');commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo).decode().strip()
assert subprocess.check_output(['git','status','--porcelain'],cwd=repo)==b''
proof=json.loads(pathlib.Path('work/search-g1-publication-86d8e76e-independent-readback.json').read_text());assert proof['published_control_commit']==commit
files=proof['files']
archive=gzip.compress(subprocess.check_output(['git','archive','--format=tar',commit],cwd=repo))
script="""import pathlib,json,os,sys,tarfile,io,hashlib,datetime,uuid,subprocess
payload=sys.stdin.buffer
size=int.from_bytes(payload.read(8),'big');meta=json.loads(payload.read(size))
P=pathlib.Path;base=P('/mnt/data/shaoshitong/search-g1-final200-bs256-20261014-v2');source=P('/mnt/data/shaoshitong/search-g1-speed-resume5-0553ade-20261012');commit=meta['commit'];files=meta['files']
old=P('/mnt/data/shaoshitong/search-g1-final200-bs256-20261014-v1')
assert json.loads((old/'logs/cleanup-status.json').read_text())['cleaned']
assert not subprocess.check_output(['squeue','-h','-w','h100-8','-o','%i'],text=True).strip()
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
def once(p,x):
 with P(p).open('x') as f:json.dump(x,f,indent=2);f.flush();os.fsync(f.fileno())
terminal={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job':3394,'cleanup':json.loads((old/'logs/cleanup-status.json').read_text()),'accounting_query':subprocess.run(['sacct','-j','3394','--format=JobID,State,ExitCode','-n'],capture_output=True,text=True).__dict__,'logs':{}}
for p in sorted((old/'logs').glob('*.log')):
 st=p.stat();terminal['logs'][p.name]={'inode':st.st_ino,'bytes':st.st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
once(old/'preparation/terminal3394-full-sha-v1.json',terminal)
base.mkdir();(base/'preparation').mkdir();(base/'logs').mkdir();(base/'journal').mkdir();(base/'repo').mkdir()
for n in ['env','bootstrap-tools','data']:(base/n).symlink_to(source/n)
import shutil
for n in ['asset-receipt.json','source-config.json','stop3381-receipt.json']:shutil.copyfile(old/'preparation'/n,base/'preparation'/n)
once(base/'preparation/deploy-intent.json',{'commit':commit,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'prior_terminal':3394})
root=base/'repo';archive=payload.read(meta['archive_bytes']);assert len(archive)==meta['archive_bytes']
with tarfile.open(fileobj=io.BytesIO(archive),mode='r:gz') as t:
 for m in t.getmembers():
  name=P(m.name);assert not name.is_absolute() and '..' not in name.parts and (m.isdir() or m.isfile())
 t.extractall(root)
for n,s in files.items():assert hashlib.sha256((root/n).read_bytes()).hexdigest()==s,n
sys.path.insert(0,str(root));from reproduction.common8.final_resume_journal import make_binding
from reproduction.common8.validation_identity import expected
r=json.loads((base/'preparation/asset-receipt.json').read_text());record,frame,ids=expected(source,'final');assert record['sha256']==r['validation_sha256']
cfg=json.loads((base/'preparation/source-config.json').read_text())['resolved_config']
identity=uuid.uuid4().hex
binding={'role':'final_only_cp200_resume','checkpoint_step':200,'checkpoint':r['checkpoint'],'checkpoint_manifest':r['checkpoint_manifest'],'checkpoint_manifest_sha256':r['checkpoint_manifest_sha256'],'generation_config_sha256':r['generation_config_sha256'],'resolved_config_sha256':r['resolved_config_sha256'],'validation_sha256':record['sha256'],'run_id':r['run_id'],'source_commit':commit,'source_sha256':files,'journal_binding':make_binding(r['run_id'],r['checkpoint_manifest_sha256'],record['sha256'],ids,frame['data_source'].tolist()),'source_completed200_run':str(source),'execution_id':identity,'batch_size':256,'vllm_gpu_memory_utilization':.35,'old_partial_rows_reused':False,'no_optimizer_or_training':True}
once(base/'preparation/final-binding.json',binding)
physical=[line.split(',')[1].strip() for line in subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True).splitlines()];assert len(physical)==len(set(physical))==8
locks=P('/mnt/data/shaoshitong/.search-g1-gpu-uuid-locks');locks.mkdir(exist_ok=True)
manifest={'role':'final_only_cp200_resume','source_completed_outer':200,'base':str(base),'world_size':8,'cpu_request':96,'memory_request':'512G','run_id':r['run_id'],'execution_id':identity,'source_base':str(source),'binding':str(base/'preparation/final-binding.json'),'config':str(base/'preparation/source-config.json'),'journal':str(base/'journal'),'physical_gpu_uuids':physical,'gpu_lock_directory':str(locks),'retrieval_url':cfg['retriever']['url'],'retriever_gpu_lib':str(source/'retriever-gpu-lib'),'retrieval_server_source':str(root/'search_r1/search/retrieval_server.py'),'retrieval_index_path':r['retrieval_index_path'],'retrieval_corpus_path':r['retrieval_corpus_path'],'retrieval_encoder':str(source/'assets/e5-base-v2'),'ray_tmp_dir':'/tmp/g1f-'+identity[:12],'preserved_foreign_gpu_processes':[],'source_commit':commit,'batch_size':256,'cache_fraction':.35}
once(base/'preparation/final-evaluation-manifest.json',manifest)
launch=base/'preparation/launch-final.sh'
launch.write_text('#!/bin/bash\\nset -euo pipefail\\nexport PATH='+str(base/'env/bin')+':/usr/local/bin:/usr/bin:/bin\\nexport PYTHONPATH='+str(root)+'\\ncd '+str(root)+'\\nexec '+str(base/'env/bin/python')+' -m reproduction.common8.final_resume_pipeline --base '+str(base)+'\\n');launch.chmod(0o700)
once(base/'preparation/deploy-receipt.json',{'commit':commit,'verified_file_count':len(files),'all_remote_full_sha_match':True,'binding':str(base/'preparation/final-binding.json'),'manifest':str(base/'preparation/final-evaluation-manifest.json'),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
print(json.dumps({'source_commit':commit,'verified_files':len(files),'manifest':str(base/'preparation/final-evaluation-manifest.json'),'batch_size':256,'cache':.35}))
"""
# The verified existing transport forwards raw source archive bytes, never weights.
sys.path.insert(0,'work');import search_g1_restore5_transport as t
original=subprocess.run
def direct(args,*a,**kw):
 if isinstance(args,list) and 'mgmt-host' in args:
  args=list(args);option='ProxyCommand=/usr/bin/nc -X 5 -x 127.0.0.1:7892 %h %p'
  if option in args:i=args.index(option);del args[i-1:i+1]
 return original(args,*a,**kw)
t.subprocess.run=direct
header=json.dumps({'commit':commit,'files':files,'archive_bytes':len(archive)}).encode()
result=t.remote(str('/mnt/data/shaoshitong/search-g1-speed-resume5-0553ade-20261012/env/bin/python')+' -c '+shlex.quote(script),'outputs/gigpo-e1-instruct200/SEARCH-G1-FINAL-BS256-DEPLOY-CORRECTED-v3.json',stdin=len(header).to_bytes(8,'big')+header+archive)
