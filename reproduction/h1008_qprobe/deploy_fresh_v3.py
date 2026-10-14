"""Once-only isolated candidate deployment; never modify any active runtime."""
import pathlib,json,gzip,subprocess,sys,shlex,datetime
repo=pathlib.Path(__file__).resolve().parents[2];commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo).decode().strip()
assert subprocess.check_output(['git','status','--porcelain'],cwd=repo)==b''
proof=json.loads(pathlib.Path(sys.argv[1]).read_text());assert proof['published_control_commit']==commit
archive=gzip.compress(subprocess.check_output(['git','archive',commit],cwd=repo));files=proof['files']
script='''import sys,json,pathlib,io,tarfile,hashlib,os,datetime,shutil
P=pathlib.Path;stream=sys.stdin.buffer;size=int.from_bytes(stream.read(8),'big');meta=json.loads(stream.read(size))
# Consume the bounded archive before subprocesses/asset checks can inherit stdin.
archive=stream.read(meta['archive_bytes']);assert len(archive)==meta['archive_bytes']
base=P(meta['base']);source=P('/mnt/data/shaoshitong/search-g1-speed-resume5-0553ade-20261012')
assert not base.exists(),'Fresh deployment already claimed; inspect receipt, never replay'
def once(p,v):
 with P(p).open('x') as f:json.dump(v,f,indent=2);f.flush();os.fsync(f.fileno())
base.mkdir();
for n in ('repo','data','preparation','logs','tmp','cache','artifacts','checkpoints'):(base/n).mkdir()
once(base/'preparation/deploy-intent.json',{'source_commit':meta['commit'],'role':'fresh_h1008_transformer3_question_probe200','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'gpu_job_submitted':False})
with tarfile.open(fileobj=io.BytesIO(archive),mode='r:gz') as t:
 for m in t.getmembers():
  name=P(m.name);assert not name.is_absolute() and '..' not in name.parts and (m.isdir() or m.isfile())
 t.extractall(base/'repo')
for n,sha in meta['files'].items():assert hashlib.sha256((base/'repo'/n).read_bytes()).hexdigest()==sha,n
for n in ('env','bootstrap-tools','bootstrap-assets','assets','retriever-gpu-lib'):
 assert (source/n).exists();(base/n).symlink_to((source/n).resolve(),target_is_directory=True)
for n in ('manifest.json','train.parquet','validation.parquet','stage.parquet','final.parquet','exclusions-audit.json'):
 assert (source/'data'/n).is_file();(base/'data'/n).symlink_to((source/'data'/n).resolve())
shutil.copyfile(source/'preparation/official-model-manifest-original.json',base/'preparation/official-model-manifest-original.json')
r=json.loads((source/'logs/run-manifest.json').read_text());assert r['model_revision']=='aa8e72537993ba99e69dfaafa59ed015b17504d1'
zero=source/'artifacts/round_0-attempt-22c55c34f646454f9c1f158ebdeb59c2/version-manifest.json';assert hashlib.sha256(zero.read_bytes()).hexdigest()=='924e5487d19cf667dee56ae0a76eba7143cd6eb6325f1024989bac5d05efbb61'
config={'role':'fresh_h1008_transformer3_question_probe200','base':str(base),'source_commit':meta['commit'],'official_model':r['initial_model'],'official_manifest':str(base/'preparation/official-model-manifest-original.json'),'retrieval_assets':r['local_retrieval_assets'],'reward_zero_source_base':str(source),'reward_zero_archive':str(zero),'reward_zero_archive_sha256':'924e5487d19cf667dee56ae0a76eba7143cd6eb6325f1024989bac5d05efbb61','predecessor_terminal_evidence_base':'/mnt/data/shaoshitong/search-g1-qprobe3-h1008-86c80e9-20261015-v2','predecessor_evaluation_base':'/mnt/data/shaoshitong/search-g1-final200-bs256-20261014-v2','predecessor_run_id':r['run_id'],'predecessor_checkpoint_manifest_sha256':'fcd9e2f87e44170dc0e9c37da8139db712a844290aa011bb130fbf39fb71aae5','predecessor_validation_sha256':'55d515e116a4c6aa63b45282f9bc87202b030156be41be60420c38a9d3ff0ed1','restore_receipt':None,'export_port':19874}
once(base/'preparation/h1008-launch-config.json',config);once(base/'preparation/source-files.json',meta['files'])
receipt={'source_commit':meta['commit'],'verified_files':len(meta['files']),'all_target_source_full_sha_match':True,'base':str(base),'gpu_job_submitted':False,'model_optimizer_rollout_transferred_to_mac':False,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
once(base/'preparation/deploy-receipt.json',receipt);print(json.dumps(receipt))
'''
sys.path.insert(0,'work');import search_g1_restore5_transport as t
original=subprocess.run
def direct(args,*a,**kw):
 if isinstance(args,list) and 'mgmt-host' in args:
  args=list(args);option='ProxyCommand=/usr/bin/nc -X 5 -x 127.0.0.1:7892 %h %p'
  if option in args:i=args.index(option);del args[i-1:i+1]
 return original(args,*a,**kw)
t.subprocess.run=direct
header=json.dumps({'commit':commit,'files':files,'archive_bytes':len(archive),'base':'/mnt/data/shaoshitong/search-g1-qprobe3-h1008-'+commit[:7]+'-20261015-v3'}).encode()
t.remote('/usr/bin/python3 -c '+shlex.quote(script),'outputs/gigpo-e1-instruct200/SEARCH-G1-QPROBE3-H1008-DEPLOY-FRESH-V3-'+commit[:8]+'.json',stdin=len(header).to_bytes(8,'big')+header+archive)
