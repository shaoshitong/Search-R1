"""One read-only published CLI replay: existing completion fresh output SHA only."""
import hashlib,json,os,pathlib,subprocess,time
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/peer-threebank58432-internal5120-plan-v1');code=b/'cpu-code-8bb6f2f6d120'
def fp(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
pub=json.loads((code/'cpu-source-publication.json').read_text());assert pub['commit']=='8bb6f2f6d1205f847b004f026d6a6432bef16289'
for name,item in pub['files'].items():assert fp(code/name)==dict(item,path=str(code/name))
original=fp(b/'complete.json');assert original['sha256']=='8a4ed0b24296ad8a5edc1bcecb249565b877d3bababb3f0221a8e7c26da72d06'
env=dict(os.environ,QWEN7B_CODE_ROOT=str(code),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1');start=time.time()
p=subprocess.run(['bash',str(code/'reproduction/operations/qwen7b_warmup/runtime.sh'),'-m','reproduction.operations.qwen7b_warmup.peer_threebank_plan','--config',str(b/'config.json')],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=40)
assert p.returncode==0,p.stderr[-1500:];assert fp(b/'complete.json')==original
value=json.loads(p.stdout.strip().splitlines()[-1])['result'];progress=[json.loads(x) for x in (b/'cpu.log').read_text().splitlines() if x.startswith('{"status": "ACTUAL_SOURCE_BANK_METADATA_ADVANCING"')]
assert [x['accepted_banks'] for x in progress]==[1,2,3]
print(json.dumps(dict(status='EXISTING_THREEBANK_COMPLETE_AND_ALL_OUTPUT_SHAS_FRESHLY_RELOADED',complete=original,exit_code=p.returncode,elapsed_seconds=time.time()-start,manifest_refs=len(json.loads((b/'features-layer26.json').read_text())),actual_source_progress=progress,questions=value['questions'],fit_questions=value['fit_questions'],holdout_questions=value['internal_holdout_questions'],existing_completion_unchanged=True,GPU_used=False,model_or_tensor_payload_loaded=False)))
