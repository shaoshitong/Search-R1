"""Prepare a new immutable balanced phase from completed Prefix20 epoch1."""
import argparse,copy,json,math,time,uuid
from pathlib import Path
from .download_model import atomic,fingerprints
from .balanced_sampling import plan,proof


def read(p):return json.loads(Path(p).read_text())


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--source-commit',required=True);a=p.parse_args();b=a.base;parent=b.parent;old=parent/'mixed-only-ft-cp2-prefix20-20261020'
    assert b.name=='balanced-rank1-from-epoch1-prefix20-20261020' and not (b/'config.json').exists()
    intent=read(b/'preparation/handoff.json');assert intent['rollback_explicitly_authorized'] and intent['old_job']=='742046'
    prior=read(old/'config.json');cp=read(old/'probe/epoch-0001.json');cp={k:v for k,v in cp.items() if k!='metrics'}
    assert cp['sha256']=='02884a3c76e88d1014c4876e1ee2ff4b4405e7f94de7b9112881fed183862724'
    assert cp['epoch']==1 and cp['completed_epoch'] and cp['cursor']==0 and cp['steps']==10464
    fresh=fingerprints(cp['path']);assert fresh['sha256']==cp['sha256'] and fresh['bytes']==cp['bytes']
    original=read(parent/'bank/groups.json');mixed=read(old/'bank/groups.json');lookup={g['qid']:g for g in mixed['train']}
    train=[]
    for original_group in original['train']:
        g=copy.deepcopy(original_group);k=sum(g['labels']);assert len(g['labels'])==5 and len(g['trajectory_ids'])==5
        if 0<k<5:
            m=lookup[g['qid']];assert g['labels']==m['labels'] and g['trajectory_ids']==m['trajectory_ids']
            g['pair_mask']=m['pair_mask']
        else:g['pair_mask']=[[False]*5 for _ in range(5)]
        assert all(not v or (g['labels'][i]==1 and g['labels'][j]==0) for i,row in enumerate(g['pair_mask']) for j,v in enumerate(row))
        train.append(g)
    assert len(train)==169028 and len(lookup)==67199
    assert len({g['qid'] for g in train})==169028
    diag=copy.deepcopy(mixed['diagnostic']);diag_lookup={g['qid']:g for g in mixed['diagnostic_mixed']}
    for g in diag:g['pair_mask']=diag_lookup[g['qid']]['pair_mask'] if g['qid'] in diag_lookup else [[False]*5 for _ in range(5)]
    assert len(diag)==512 and {g['qid'] for g in train}.isdisjoint(g['qid'] for g in diag)
    assert len({tid for g in train for tid in g['trajectory_ids']})==169028*5
    for name in ['bank','probe','logs','tmp','cache','preparation']:(b/name).mkdir(exist_ok=True)
    atomic(b/'bank/groups.json',dict(train=train,diagnostic=diag,diagnostic_mixed=[g for g in diag if 0<sum(g['labels'])<5]))
    plans={str(e):proof(train,plan(train,1729,e,67198)) for e in range(2,6)}
    assert all(v['questions']==67198 and v['mixed']==33599 and v['nonmixed']==33599 and v['updates']==1318 and v['unique_questions']==67198 for v in plans.values())
    c=dict(base=str(b),run_id=uuid.uuid4().hex,previous_phase=str(old),training_base=str(parent),selection_base=prior['selection_base'],source_checkpoint=cp,source_steps=10464,source_epoch=1,source_commit=a.source_commit,model_dir=prior['model_dir'],question_max_tokens=273,prefix_tokens=20,capture_prefix_tokens=20,feature_layer=26,native_head_layer=27,fit_epochs=5,fit_start_epoch=2,fit_epoch_groups=67198,fit_lr=5e-6,fit_ema=.99,pair_weight=1.,pair_margin=1.,pair_normalization='mean over Mixed questions in global batch, shared fraction across rank/microbatch',fit_world=4,fit_global_groups=51,fit_micro_groups=2,seed=1729,train_pool_questions=169028,sampling='1:1 Mixed/nonMixed; sample67198 complete questions without replacement per epoch',actor_updates_authorized=False,mixed_bank_sha256=fingerprints(b/'bank/groups.json')['sha256'],fit_file_sha256=fingerprints(Path(__file__).with_name('balanced_finetune.py'))['sha256'],previous_fit_file_sha256=prior['fit_file_sha256'],source_config_sha256=fingerprints(old/'config.json')['sha256'],source_bank_sha256=fingerprints(parent/'bank/complete.json')['sha256'])
    atomic(b/'config.json',c)
    report=dict(status='BALANCED_RANK1_PHASE_PREPARED_FROM_SEALED_EPOCH1',source_checkpoint=cp,source_fresh_sha256=fresh['sha256'],source_commit=a.source_commit,pair_weight_before=.1,pair_weight_after=1.,sampling_plans=plans,remaining_epochs=[2,3,4,5],final_target_steps=15736,train_diagnostic_questions_disjoint=True,raw_EMA_Adam_RNG_continuation_required=True,actor_updates=0,time=time.time())
    atomic(b/'preparation/acceptance.json',report);print(json.dumps(report))


if __name__=='__main__':main()
