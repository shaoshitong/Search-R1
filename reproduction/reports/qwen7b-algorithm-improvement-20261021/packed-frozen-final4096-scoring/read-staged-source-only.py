import pathlib,json,hashlib,time
base=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/packed-variance-frozen-final4096-20261023-v1')
def proof(p):
 raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
pub=json.loads((base/'source-publication.json').read_bytes());assert pub['commit']=='67dc1a3509a3af82de0d4f8b17cca21e1216acaf';code=pathlib.Path(pub['code_root']);files={}
for name,ref in pub['files'].items():
 actual=proof(code/name);assert all(actual[k]==ref[k]for k in('bytes','sha256'));files[name]=actual
staged=json.loads((base/'staged.json').read_bytes());template=json.loads((base/'candidate-freeze-TEMPLATE.json').read_bytes());assert template['status']=='TEMPLATE_ONLY_NOT_CANDIDATE_FREEZE'and staged['candidate_frozen']is False and staged['Slurm_submitted']is False
print(json.dumps(dict(time=time.time(),status='PACKED_FINAL_SCORER_SOURCE_REFS_FRESH_SHA_PASS_WITHOUT_FINAL_READ',publication=proof(base/'source-publication.json'),entry=proof(base/'scoring.sbatch'),files=files,CPU_import_passed=staged['CPU_import_passed'],candidate_frozen=False,final_data_or_model_opened=False,Slurm_submitted=False)))
