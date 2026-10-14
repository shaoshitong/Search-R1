"""One read of actual prepared qprobe JSON/source bytes; no assets or GPU calls."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import tempfile
import time

ROOT=Path('/Users/shaoshitong/Documents/Search-G1')
REMOTE=r'''
import hashlib,json,os,re,socket,subprocess,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
payload=json.load(sys.stdin);intent=payload['intent'];base=Path(intent['base']);repo=base/'repo'
def read(path):
 path=Path(path);assert path.resolve()==path and path.is_file() and path.stat().st_uid==os.getuid()
 assert path.stat().st_size<=1024**2
 return json.loads(path.read_text())
def sha_doc(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
assert intent['role']=='a800_qprobe_prospective_intent' and intent['host']=='baai-a800'
assert re.fullmatch('[0-9a-f]{32}',intent['deployment_id'])
assert str(base)=='/data/shaoshitong/Search-G1-a800-fresh200-'+intent['deployment_id']
assert read(base/'preparation/qprobe-intent.json')==intent
assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True,timeout=10).strip()==intent['source_commit']
assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],timeout=10).strip()
sys.path.insert(0,str(repo))
from reproduction.common8.source_files import runtime_files
source={str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in runtime_files(repo)}
assert source==intent['source_sha256']
run=read(base/'logs/run-manifest.json')
assert run['run_id']==payload['run_id'] and run['execution_id']==payload['execution_id']
assert run['source_commit']==intent['source_commit'] and run['source_sha256']==source
assert run['execution_backend']=='ssh_local' and run['execution_host']==socket.gethostname() and run['slurm_job_id'] is None
assert run['completed_updates']==0 and run['training_world_size']==4 and run['gpu_placement_bound'] is False
fields=('run_id','execution_id','source_commit','execution_backend','execution_host','completed_updates','target_updates',
 'contract_kind','policy_safety_mode','precision_profile','behavior_probability_policy','training_world_size',
 'actor_micro_batch_per_rank','gpu_placement_policy','gpu_placement_bound','auxiliary_physical_indices',
 'initial_model','reference_model','initial_model_sha256','model_revision','input_sha256','budget','infra_rollout','question_probe',
 'checkpoint_dir','train_log','pretraining_baseline_policy','retrieval_asset_paths','retrieval_encoder')
inherit=('full_state_restore','actor_checkpoint','reference_checkpoint','resume_path','inherited_policy_run_id',
 'inherited_optimizer_run_id','inherited_scheduler_run_id')
selected={key:run.get(key) for key in fields};selected['inherited_fields']={key:run.get(key) for key in inherit}
assert all(value is None for value in selected['inherited_fields'].values())
inputs=read(base/'logs/deployment-input-receipt.json');origin=read(base/'logs/deployment-source-receipt.json')
assert origin['source_commit']==intent['source_commit'] and origin['source_sha256']==source
assert inputs['exact_original_remote_bytes'] is True and inputs['training_data_partitioned'] is False
features=read(base/'preparation/features-ready.json')
assert features['run_id']==run['run_id'] and features['source_commit']==run['source_commit']
assert features['cache_dir']==run['question_probe']['cache_dir']
names=('pipeline-launch-claim.json','training-launch-claim.json','process-ledger.json','actual-attempts.jsonl','actual-updates.jsonl')
claims={name:(base/'logs'/name).exists() for name in names};assert not any(claims.values())
out=dict(role='actual_qprobe_prelaunch_CPU_JSON_source_read',base=str(base),source_bytes_equal=True,runtime_files=len(source),
 source_map_sha256=sha_doc(source),run=selected,features_ready=features,
 source_receipt={key:origin.get(key) for key in ('source_commit','source_transport','source_is_exact_published_commit','uncommitted_source_copied')},
 input_receipt=inputs,execution_claims=claims,model_or_gpu_loaded=False,read_once=True,remote_mutation=False,checked_unix=time.time())
raw=json.dumps(out,allow_nan=False);assert len(raw.encode())<128*1024
print('PRELAUNCH_JSON:'+raw,flush=True)
'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--handle',type=Path,required=True);p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--run-id',required=True);p.add_argument('--execution-id',required=True)
    p.add_argument('--receipt',type=Path,required=True);args=p.parse_args()
    assert all(re.fullmatch('[0-9a-f]{32}',x) for x in (args.run_id,args.execution_id))
    args.handle.relative_to(ROOT/'preparation');args.receipt.relative_to(ROOT/'preparation')
    assert not args.receipt.exists();intent=json.loads(args.handle.read_text())
    spec=importlib.util.spec_from_file_location('runtime_deploy_qprobe',args.runtime/'reproduction/a800_common/deploy_qprobe.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.verified_launcher(args.runtime,intent);module.validate_intent(intent);ast.parse(REMOTE)
    route=module.ssh_command(REMOTE);assert route[-1].startswith('python3 -c ')
    route[-1]='/data/search-g1/env/bin/python -c '+route[-1][len('python3 -c '):]
    value=dict(role='single_actual_prelaunch_metadata_receipt',handle=str(args.handle),run_id=args.run_id,
        execution_id=args.execution_id,source_commit=intent['source_commit'],remote_code_sha256=hashlib.sha256(REMOTE.encode()).hexdigest())
    payload=json.dumps(dict(intent=intent,run_id=args.run_id,execution_id=args.execution_id)).encode()
    with tempfile.TemporaryFile() as out,tempfile.TemporaryFile() as err:
        r=subprocess.run(route,input=payload,stdout=out,stderr=err,timeout=90)
        out.seek(0);raw=out.read(128*1024+1);err.seek(0)
        value.update(exit_code=r.returncode,stdout_sha256=hashlib.sha256(raw).hexdigest(),stderr_sha256=hashlib.sha256(err.read(65536)).hexdigest())
    value['observation']=None
    if r.returncode==0 and len(raw)<=128*1024:
        rows=[json.loads(x.split(':',1)[1]) for x in raw.decode().splitlines() if x.startswith('PRELAUNCH_JSON:')]
        assert len(rows)==1;value['observation']=rows[0]
    value['observed_unix']=time.time()
    module.write_once(args.receipt,value)
    print(json.dumps(dict(receipt=str(args.receipt),exit_code=r.returncode,observed=bool(value['observation']))))
    return r.returncode


if __name__=='__main__':raise SystemExit(main())
