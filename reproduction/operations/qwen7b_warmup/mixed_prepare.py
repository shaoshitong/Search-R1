"""Prepare immutable real-Mixed continuation bank. Does not submit GPU jobs."""
import argparse,gzip,json,hashlib,math,os,subprocess,time,uuid
from pathlib import Path
from .download_model import atomic,fingerprints

def read(p):return json.loads(Path(p).read_text())
def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--source-commit',required=True);p.add_argument('--prefix-tokens',type=int,choices=(5,20),default=5);a=p.parse_args()
    b=a.base;source=b.parent;prefix=a.prefix_tokens;assert b.name==('mixed-only-ft-cp2-prefix20-20261020' if prefix==20 else 'mixed-only-ft-cp2-20261020')
    assert not (b/'config.json').exists(),'Existing immutable phase: inspect, do not recreate'
    original=read(source/'config.json');all_groups=read(source/'bank/groups.json');cp=read(source/'probe/epoch-0002.json')
    assert cp['sha256']=='893e2d71e9b45abb92a9573a6091a1a0fb0a5f7ffd3e4e1201d12061317095ad'
    assert fingerprints(cp['path'])['sha256']==cp['sha256'] and cp['steps']==9146
    train=[g for g in all_groups['train'] if 0<sum(g['labels'])<5];diag=[g for g in all_groups['diagnostic'] if 0<sum(g['labels'])<5]
    assert len(train)==67199 and len(diag)==205
    needed={g['qid']:g for g in train+diag};tokens={};paths=set()
    for feature in read(source/'bank/features.json'):paths.update(Path(feature).parent.parent.glob('rank-*-prefix.jsonl.gz'))
    for path in sorted(paths):
        with gzip.open(path,'rt') as f:
            for line in f:
                r=json.loads(line)
                if r['qid'] in needed:tokens.setdefault(r['qid'],{})[r['response_index']]=(r['trajectory_id'],tuple(r['policy_token_ids'][:prefix]),tuple(r['policy_position_ids'][:prefix]))
    total=eligible=groups_without=0
    for qid,g in needed.items():
        assert set(tokens[qid])==set(range(5));own=[tokens[qid][i] for i in range(5)]
        assert [t[0] for t in own]==g['trajectory_ids']
        g['pair_mask']=[[bool(g['labels'][i]==1 and g['labels'][j]==0 and own[i][1:]!=own[j][1:]) for j in range(5)] for i in range(5)]
        n=sum(map(sum,g['pair_mask']));groups_without+=n==0;eligible+=n;total+=sum(g['labels'])*(5-sum(g['labels']))
    for name in ('bank','probe','logs','tmp','cache','preparation'):(b/name).mkdir(parents=True,exist_ok=True)
    atomic(b/'bank/groups.json',dict(train=train,diagnostic=all_groups['diagnostic'],diagnostic_mixed=diag))
    c=dict(base=str(b),run_id=uuid.uuid4().hex,training_base=str(source),selection_base=str(source/'selection-2048-20261020'),source_checkpoint=cp,source_commit=a.source_commit,model_dir=original['model_dir'],question_max_tokens=273,prefix_tokens=prefix,source_prefix_tokens=5,capture_prefix_tokens=20,feature_layer=26,native_head_layer=27,fit_epochs=5,fit_lr=5e-6,fit_ema=.99,pair_weight=.1,pair_margin=1.,fit_world=4,fit_global_groups=51,fit_micro_groups=2,seed=1729,train_questions=67199,train_trajectories=335995,sampling='unique_genuine_Mixed_once_each_epoch_no_replacement',pair_eligibility='same_question_positive_negative_different_policy%d_tokens_or_positions'%prefix,actor_updates_authorized=False,mixed_bank_sha256=fingerprints(b/'bank/groups.json')['sha256'],fit_file_sha256=fingerprints(Path(__file__).with_name('mixed_finetune.py'))['sha256'],source_config_sha256=fingerprints(source/'config.json')['sha256'],source_bank_sha256=fingerprints(source/'bank/complete.json')['sha256'])
    atomic(b/'config.json',c)
    report=dict(status='CP2_REAL_MIXED_ONLY_BANK_PREPARED',prefix_tokens=prefix,questions=67199,diagnostic_questions=512,diagnostic_mixed_questions=205,total_positive_negative_pairs_train_and_diag=total,eligible_pairs_train_and_diag=eligible,groups_without_distinguishable_pair_train_and_diag=groups_without,epochs=5,unique_groups_per_epoch=67199,steps_per_epoch=math.ceil(67199/51),source_checkpoint=cp,config_sha256=fingerprints(b/'config.json')['sha256'],bank_sha256=c['mixed_bank_sha256'],source_commit=a.source_commit,time=time.time())
    atomic(b/'preparation/acceptance.json',report);print(json.dumps(report),flush=True)
if __name__=='__main__':main()
