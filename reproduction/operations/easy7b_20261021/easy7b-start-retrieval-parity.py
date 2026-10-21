import json,time,subprocess,os,hashlib,shlex
from pathlib import Path
s=Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-retrieval-service2-20261021');old=Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-20261021');commit='f76207ec430700b6f7004037e4378cbc3fce3664';src=s/'source'/commit
import sys;sys.path.insert(0,str(src))
from reproduction.easy2560_joint4.control import publish,sha
from reproduction.common8.http_retrieval import compare_response
oldproof=json.loads((old/'logs/retrieval-admission.json').read_text());newproof=json.loads((s/'logs/stage64-admission.json').read_text());ready=json.loads((s/'logs/service-ready.json').read_text())
assert oldproof['stage_sha256']==newproof['stage_sha256'] and oldproof['question_ids']==newproof['question_ids']
assert oldproof['gpu_document_ids']==newproof['gpu_document_ids']
assert oldproof['gpu_document_sha256']==newproof['gpu_document_sha256']
errors=[abs(a-b) for x,y in zip(oldproof['gpu_document_scores'],newproof['gpu_document_scores']) for a,b in zip(x,y)]
assert max(errors)<=1e-4
record=dict(accepted=True,epoch=time.time(),scope='actual four-physical versus two-physical/four-logical GPU stage64 ordered document IDs/hashes and scores',query_count=64,document_count=192,ordered_document_ids_equal=True,ordered_document_hashes_equal=True,score_max_abs_error=max(errors),score_mean_abs_error=sum(errors)/len(errors),old_proof_sha256=sha(old/'logs/retrieval-admission.json'),new_proof_sha256=sha(s/'logs/stage64-admission.json'),service_job_id='745198',old_job_id='745022')
p=s/'diagnostics/stage64-four-vs-two.json'
if not p.exists():publish(p,record)
else:record=json.loads(p.read_text())
root=s/'diagnostics/actor-visible-parity';root.mkdir(parents=True,exist_ok=True);claim=root/'launch.json'
if claim.exists():print(json.dumps(dict(stage64=record,existing_claim=json.loads(claim.read_text()))));raise SystemExit(0)
raw=old/'artifacts/mixed-candidate-pools/857092f9bb234821ab62664bf273a1f0-pool-0/0.pt'
model=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/assets/qwen25-7b-instruct-a09a35458c70')
cmd=['srun','--jobid=745198','--overlap','--exact','--ntasks=1','--cpus-per-task=2','--mem=8G','--gres=none','env','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=2',str(s/'env/bin/python'),'-B',str(src/'reproduction/operations/easy7b_20261021/compare_saved_observations.py'),'--raw-pool-file',str(raw),'--service-ready',str(s/'logs/service-ready.json'),'--tokenizer',str(model),'--output',str(root/'result.json'),'--max-cases','64']
publish(claim,dict(epoch=time.time(),source_commit=commit,command=cmd,source_file_sha256=sha(src/'reproduction/operations/easy7b_20261021/compare_saved_observations.py'),separate_gpu_allocation=False))
wrapper='import subprocess,json,time,pathlib; r=subprocess.run('+repr(cmd)+'); pathlib.Path('+repr(str(root/'exit.json'))+').write_text(json.dumps(dict(returncode=r.returncode,completed_unix=time.time()))+"\\n")'
with (root/'stdout.log').open('xb')as f:p=subprocess.Popen([str(s/'env/bin/python'),'-B','-c',wrapper],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(stage64=record,parity_pid=p.pid,parity_root=str(root))))
