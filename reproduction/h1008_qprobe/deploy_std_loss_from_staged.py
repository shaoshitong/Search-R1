"""Reuse only staged Git source and canonical input links after pre-init failure."""
import datetime,hashlib,json,subprocess,sys
from pathlib import Path
from reproduction.h1008_qprobe.management_transport import remote

REMOTE='''import pathlib,json,hashlib,shutil,os,subprocess,time
P=pathlib.Path;meta=META
old=P('/mnt/data/shaoshitong/search-g1-qprobe3-stdloss-38047c0-20261015-v1');base=P(meta['base']);source=P('/mnt/data/shaoshitong/search-g1-qprobe3-h1008-003a0b0-20261015-v3')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def once(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2);f.flush();os.fsync(f.fileno())
assert not base.exists(),'Existing deployment: inspect, never replay'
assert not (old/'logs/run-manifest.json').exists() and not (old/'logs/actual-updates.jsonl').exists()
native=subprocess.check_output(['scontrol','show','job','-o','3398'],text=True)
assert 'JobId=3398 ' in native and 'JobState=FAILED' in native and 'ExitCode=1:0' in native
assert 'AssertionError: Existing derivative must not be silently replaced' in (old/'logs/native-3398.log').read_text()
assert not list(P('/sys/fs/cgroup').glob('**/job_3398'))
if not (old/'preparation/failed3398-terminal.json').exists():once(old/'preparation/failed3398-terminal.json',dict(native=native,native_log_sha256=sha(old/'logs/native-3398.log'),actual_actor_updates=0,model_initialization_started=False,checked_unix=time.time()))
base.mkdir()
for d in ('data','preparation','logs','tmp','cache','artifacts','checkpoints'):(base/d).mkdir()
once(base/'preparation/deploy-intent.json',dict(source_commit=meta['commit'],failed_preinit_job=3398,original_directory_retained=True))
shutil.copytree(old/'repo',base/'repo',ignore=shutil.ignore_patterns('__pycache__'))
files=json.loads((old/'preparation/source-files.json').read_text())
for name,text in meta['changes'].items():
 p=base/'repo'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text);files[name]=hashlib.sha256(text.encode()).hexdigest()
assert len(files)==meta['file_count'] and hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest()==meta['manifest_sha256']
for name,digest in files.items():assert sha(base/'repo'/name)==digest,name
for d in ('env','bootstrap-tools','bootstrap-assets','assets','retriever-gpu-lib'):(base/d).symlink_to((source/d).resolve(),target_is_directory=True)
for name in ('manifest.json','train.parquet','validation.parquet','stage.parquet','final.parquet','exclusions-audit.json'):(base/'data'/name).symlink_to((source/'data'/name).resolve())
for name in ('official-model-manifest-original.json','predecessor-exit0.json','predecessor-teardown-closure.json','std-loss-old-terminal.json'):
 (base/'preparation'/name).write_bytes((old/'preparation'/name).read_bytes())
config=json.loads((old/'preparation/h1008-launch-config.json').read_text());config.update(base=str(base),source_commit=meta['commit'],official_manifest=str(base/'preparation/official-model-manifest-original.json'))
(base/'question-features').symlink_to(P(config['precomputed_question_features']['path']),target_is_directory=True)
once(base/'preparation/source-files.json',files);once(base/'preparation/h1008-launch-config.json',config)
receipt=dict(base=str(base),source_commit=meta['commit'],verified_files=len(files),all_target_source_full_sha_match=True,submitted=False,canonical_inputs_only=True,old_models_or_optimizers_inherited=False)
once(base/'preparation/deploy-receipt.json',receipt);print(json.dumps(receipt))
'''


def main():
    repo=Path(__file__).resolve().parents[2];proof=json.loads(Path(sys.argv[1]).read_text())
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo).decode().strip()
    assert proof['published_control_commit']==commit and proof['all_blobs_full_sha_match']
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo)
    names=subprocess.check_output(['git','diff','--name-only','38047c0d',commit],cwd=repo).decode().splitlines()
    changes={n:(repo/n).read_text() for n in names}
    files=proof['files'];meta=dict(commit=commit,changes=changes,file_count=len(files),manifest_sha256=hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest(),base='/mnt/data/shaoshitong/search-g1-qprobe3-stdloss-'+commit[:7]+'-20261015-v2')
    socket=json.loads(Path('work/qprobe3-management-current.json').read_text())['socket']
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    print(remote(REMOTE.replace('META',repr(meta)),'work/qprobe-stdloss-deploy-v2-'+stamp+'.json',socket))


if __name__=='__main__':main()
