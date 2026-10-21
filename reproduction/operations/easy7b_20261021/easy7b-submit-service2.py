import os,sys,subprocess,json,time,shlex,uuid
from pathlib import Path
base=Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-retrieval-service2-20261021');commit='262428b0ecce045a60b2d2cd054e3b906673df8c';old=Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-20261021')
sys.path.insert(0,str(base/'source'/commit))
from reproduction.easy2560_joint4.control import bootstrap,publish
bootstrap(base,commit)
claim=base/'preparation/service-submit-claim.json'
if claim.exists() or (base/'job.id').exists():raise RuntimeError('Existing service claim: inspect actual job, do not submit twice')
service_run=uuid.uuid4().hex
libraries=list((base/'env/lib/python3.10/site-packages/nvidia').glob('*/lib'))+[base/'env/lib/python3.10/site-packages/torch/lib']
assets=Path('/data/user/jxu957/experiments/cosearch-continue300-x203-20261008/source/assets-root/assets/wiki')
env=dict(PATH=str(base/'env/bin')+':/usr/local/bin:/usr/bin:/bin',PYTHONPATH=str(base/'retriever-gpu-lib')+':'+str(base/'repo'),LD_LIBRARY_PATH=':'.join(map(str,libraries)),HF_HOME=str(base/'cache/hf'),XDG_CACHE_HOME=str(base/'cache'),TMPDIR=str(base/'tmp'),OMP_NUM_THREADS='8',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',SEARCH_G1_BASE=str(base),SEARCH_G1_SOURCE_COMMIT=commit)
program=['env',*['%s=%s'%x for x in env.items()],str(base/'env/bin/python'),'-B','-m','reproduction.easy2560_joint4.retrieval_service','--base',str(base),'--run-id',service_run,'--training-run-id','232f20e112ac4a978c5aba52a0367195','--source-commit',commit,'--index',str(assets/'e5_Flat.index'),'--corpus',str(assets/'wiki-18.jsonl'),'--encoder',str(base/'assets/e5-base-v2'),'--stage-parquet',str(old/'data/stage.parquet'),'--host','0.0.0.0','--port','18020']
command=['sbatch','--parsable','--partition=emergency_acd','--qos=formal-user','--account=jxu957','--nodes=1','--ntasks=1','--gres=gpu:2','--cpus-per-task=24','--mem=384G','--time=7-00:00:00','--job-name=easy7b-retrieval2','--export=NONE','--chdir='+str(base/'repo'),'--output='+str(base/'logs/service-%j.log'),'--error='+str(base/'logs/service-%j.log'),'--wrap=exec '+shlex.join(program)]
publish(claim,dict(epoch=time.time(),source_commit=commit,service_run_id=service_run,command=command,authorization='User2026-10-21 separate2H100 retrieval +4H100training; preserve original4logicalshards'))
r=subprocess.run(command,capture_output=True,text=True,timeout=30);record=dict(epoch=time.time(),returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,source_commit=commit,service_run_id=service_run)
if r.returncode==0:
 job=r.stdout.strip().split(';')[0];assert job.isdecimal();record['job']=job
 with (base/'job.id').open('x')as f:f.write(job+'\n');f.flush();os.fsync(f.fileno())
publish(base/'preparation/service-submission.json',record);print(json.dumps(record));sys.exit(bool(r.returncode))
