from pathlib import Path
import hashlib,json,time
b=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/frozen-final4096-full169028-L15P80-MSE-20261023-v2')
def proof(p):
 raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
staged=json.loads((b/'staged.json').read_bytes());pub=json.loads((b/'source-publication.json').read_bytes());assert proof(b/'source-publication.json')==staged['publication'];files={}
for rel,ref in pub['files'].items():
 p=Path(pub['code_root'])/rel;actual=proof(p);assert actual==dict(path=str(p),**ref);files[rel]=actual
wrapper=files['reproduction/operations/qwen7b_warmup/score_frozen_full169028_MSE_final4096.py'];assert wrapper['sha256']=='65d853fefd922b85ec723c9194b8e1d823703652925059a8d74e898e2349d791';assert proof(b/'scoring.sbatch')==staged['entry'];assert proof(b/'candidate-freeze-TEMPLATE.json')==staged['template'];template=json.loads((b/'candidate-freeze-TEMPLATE.json').read_bytes());assert template['status']=='FUTURE_ONLY_NOT_FROZEN'and template['freeze_epoch']is None and template['configuration']['epoch']is None and template['configuration']['primary_state']is None and all(r['init_seed']is None and r['train_plan_seed']is None for r in template['replications']);assert not(b/'final-use-started.json').exists()and not(b/'complete.json').exists();print(json.dumps(dict(status='FULL169028_FUTURE_SCORER_SOURCE_CLOSURE_AND_ENTRY_INDEPENDENTLY_REHASHED',time=time.time(),source_publication=proof(b/'source-publication.json'),source_files=files,entry=proof(b/'scoring.sbatch'),template=proof(b/'candidate-freeze-TEMPLATE.json'),actual_candidate_frozen=False,final_inputs_labels_or_checkpoints_opened=False,GPU_or_Slurm_action=False)))
