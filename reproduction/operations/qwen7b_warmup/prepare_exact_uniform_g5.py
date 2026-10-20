"""CPU typed input preparation for ONE authorized same-source G10 all252 fork."""
import argparse,json,time
from pathlib import Path
from . import probe_ablation_train as u
from . import probe_g10_train as inherited
from .probe_exact_uniform_g5_train import make_config

REFERENCE_SHA='224558c829cb4b1d63330d0008d3fdaa57acb2cba624cc3fb56cb9b9deba49ec'
PLAN_SHA='3bff8f5331bd60634456b1f24a106a752f84b196bc0a0cd8961c5d17405e86d9'
TRAIN_SHA='f158668aa439fcb8735a0e311939cc8340d40aa4c44bab8a4a85bbb9bf88d7af'
VALIDATION_SHA='5ea5197a841d7bd0b226a82350a2436a8ae3d860143a2ea39fe59b7ecb6837d5'
ROOT=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021')
REFERENCE=ROOT/'g10-matched6528-v1/random/config.json'
TARGET=ROOT/'g10-exact-uniform252-v1'


def ref(p):
    p=Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=u.sha(p))


def prepare(base):
    if base!=TARGET or base.resolve()!=base:raise ValueError('Independent authorized all252 root required')
    if (base/'preparation.json').exists():return u.read(base/'preparation.json')
    if u.sha(REFERENCE)!=REFERENCE_SHA:raise ValueError('Actual pinned random4 reference configuration changed')
    reference=u.read(REFERENCE);plan=u.read(u.checked(reference['plan']))
    if reference['plan']['sha256']!=PLAN_SHA or reference['g10_complete']['sha256']!=TRAIN_SHA or reference['validation_g10_complete']['sha256']!=VALIDATION_SHA:raise ValueError('Exact actual plan/bank/validation bindings required')
    for key in ('source_checkpoint','source_config','source_bank','model_config','g10_complete','validation_g10_complete'):u.checked(reference[key])
    groups,manifests,accepted=inherited.accepted_g10(reference['g10_complete'])
    if len(groups)!=6528 or accepted['actor_updates']!=0:raise ValueError('All real6528/65280 accepted own-EM G10 required')
    external=set();hashes=set()
    for evaluation in reference['evaluations']:
        if evaluation['role']!='validation':raise ValueError('No final evaluation in training entry')
        for key in ('bank','features','fixed_g5'):u.checked(evaluation[key])
        source=u.read(evaluation['bank']['path'])[evaluation['split']];frozen=inherited.frozen_validation(source,u.read(evaluation['fixed_g5']['path']))
        if len(frozen) not in (512,2048):raise ValueError('Only identical original diagnostic512/new fixed2048 accepted')
        qs={g['qid'] for g in frozen};hs={g['question_hash'] for g in source}
        if external&qs or hashes&hs:raise ValueError('Repeated evaluation question identity')
        external|=qs;hashes|=hs
    if len(external)!=2560 or {g['qid'] for g in groups}&external or {g['question_hash'] for g in groups}&hashes:raise ValueError('Exact train/eval isolation required')
    pub=u.read(base/'source-publication.json');code=Path(pub['code_root']);code_hashes={n:v['sha256'] for n,v in pub['files'].items()}
    for n,h in code_hashes.items():
        if u.sha(code/n)!=h:raise ValueError('Immutable published runtime changed')
    config=make_config(reference,plan,hashes=code_hashes,commit=pub['commit'],launch_authorized=True);inherited.validate_plan(plan,groups,config,external)
    config['reference_config']=ref(REFERENCE);config['execution_authorization']='Parent/root explicit GO after matched G10random4 full136 evidence; supersedes preliminary wait rule'
    p=base/'config.json'
    if p.exists():
        if u.read(p)!=config:raise ValueError('Existing immutable all252 configuration differs')
    else:u.atomic(p,config)
    result=dict(status='ACTUAL_ALL252_E1_SAME_G10_B48X136_TYPED_INPUTS_PREPARED_NOT_SUBMITTED',config=ref(p),source_publication=ref(base/'source-publication.json'),reference_config=ref(REFERENCE),plan=reference['plan'],g10_complete=reference['g10_complete'],validation_g10_complete=reference['validation_g10_complete'],source_checkpoint=reference['source_checkpoint'],questions=6528,real_trajectories=65280,global_original_questions=48,updates=136,source_Adam_steps=10464,optimizer_reset=False,new_trajectories=0,full_checkpoint_and_first_update_replay_required=True,complete_same_source_evaluation_protocol_preserved=True,model_or_checkpoint_tensor_loaded=False,feature_tensor_loaded=False,actor_updates=0,time=time.time())
    u.atomic(base/'preparation.json',result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);a=p.parse_args()
    try:result=prepare(a.base)
    except BaseException as e:u.atomic(a.base/'cpu-prepare-failure.json',dict(error_type=type(e).__name__,error=str(e),time=time.time()));raise
    print(json.dumps(result),flush=True)
