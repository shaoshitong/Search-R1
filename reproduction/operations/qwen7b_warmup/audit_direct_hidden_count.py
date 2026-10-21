"""Read-only fixed-G5 own-EM/direct-count milestone acceptance, CPU only."""
import argparse,gzip,json,math
from pathlib import Path
from . import peer_append_train as a


def metrics(rows):
    counts=[sum(r['strict_em']) for r in rows];truth=[0<k<5 for k in counts]
    score=[r['predicted_mixed'] for r in rows];pred=[q>=.5 for q in score]
    tp=sum(p and y for p,y in zip(pred,truth));fn=sum(not p and y for p,y in zip(pred,truth))
    tn=sum(not p and not y for p,y in zip(pred,truth));fp=sum(p and not y for p,y in zip(pred,truth))
    tpr=tp/(tp+fn) if tp+fn else None;tnr=tn/(tn+fp) if tn+fp else None
    positives=sum(truth);negatives=len(rows)-positives;auc=None
    if positives and negatives:
        ordered=sorted(zip(score,truth));rank_sum=0.;i=0
        while i<len(ordered):
            j=i+1
            while j<len(ordered) and ordered[j][0]==ordered[i][0]:j+=1
            rank_sum+=sum(int(y) for _,y in ordered[i:j])*(i+1+j)/2;i=j
        auc=(rank_sum-positives*(positives+1)/2)/(positives*negatives)
    return dict(questions=len(rows),mixed_questions=positives,mixed_fraction=positives/len(rows),
        mixed_balanced_accuracy=(tpr+tnr)/2 if tpr is not None and tnr is not None else None,
        mixed_recall=tpr,nonmixed_recall=tnr,mixed_roc_auc=auc,
        allwrong_FPR=sum(p and k==0 for p,k in zip(pred,counts))/counts.count(0) if 0 in counts else None,
        allcorrect_FPR=sum(p and k==5 for p,k in zip(pred,counts))/counts.count(5) if 5 in counts else None,
        confusion=dict(tp=tp,fn=fn,tn=tn,fp=fp),k_histogram=[counts.count(k) for k in range(6)])


def accept_rows(rows,frozen):
    if len(rows)!=len(frozen) or len({r['qid'] for r in rows})!=len(rows):raise ValueError('Fixed question coverage differs')
    by={g['qid']:g for g in frozen}
    for r in rows:
        g=by[r['qid']]
        if (r['trajectory_ids']!=g['trajectory_ids'] or r['strict_em']!=g['labels']
                or r.get('score_kind')!='hidden_direct_count6'):raise ValueError('Own fixedG5 IDs/strictEM/direct-count kind differs')
        p=r['p_j'];counts=r['count_probabilities'];q=r['predicted_mixed']
        if (len(p)!=5 or len(counts)!=6 or any(not isinstance(v,(float,int)) or not math.isfinite(v) or not 0<=v<=1 for v in p+counts+[q])
                or abs(sum(counts)-1)>1e-5 or abs(q-sum(counts[1:5]))>1e-6):
            raise ValueError('Six count probabilities/q_direct invalid; old p-product is not this score')
    selected=sorted(rows,key=lambda r:(-r['predicted_mixed'],str(r['qid'])))[:512]
    return dict(pool=metrics(rows),selected512=metrics(selected))


def compare(recomputed,recorded):
    for scope in ('pool','selected512'):
        for key in ('questions','mixed_questions','mixed_fraction','mixed_balanced_accuracy','mixed_recall','nonmixed_recall','mixed_roc_auc','k_histogram','confusion'):
            a1=recomputed[scope][key];b=recorded[scope][key]
            if isinstance(a1,float):
                if abs(a1-b)>1e-10:raise ValueError('Independent fixed metric differs: '+key)
            elif a1!=b:raise ValueError('Independent fixed metric differs: '+key)


def audit(base,cursor):
    from . import direct_hidden_checkpoint as cp
    from .probe_g10_train import frozen_validation
    base=Path(base)
    if not str(base).startswith(str(a.A)+'/direct-hidden-count30016-train-') or cursor not in (256,512,768,938):
        raise ValueError('Only named remote joint milestones accepted')
    config=a.proof(base/'config.json');c=a.read(base/'config.json');directory=base/'evaluations'/('cursor-%06d'%cursor)
    receipt=a.read(directory/'complete.json')
    if (receipt['cursor']!=cursor or receipt['legacy_steps']!=10464+cursor or receipt['new_head_steps']!=cursor
            or receipt['config_sha256']!=config['sha256'] or receipt['training_nine_components_unchanged'] is not True
            or receipt['before_nine_component_hashes']!=receipt['after_nine_component_hashes'] or receipt['final_pool_read'] is not False):
        raise ValueError('Joint checkpoint-first state/clock/evaluation receipt differs')
    binding=dict(config=config,source_checkpoint=c['source_checkpoint'],source_config=c['source_config'],
        source_bank=c['source_bank'],g10_complete=c['g10_complete'],plan=c['plan'],plan_sha256=c['plan']['sha256'],recipe=c['recipe'])
    verified=cp.verify_joint_file(a.checked(receipt['checkpoint']),expected_binding=binding,expected_cursor=cursor)
    if verified['accepted_hashes']!=receipt['before_nine_component_hashes']:raise ValueError('Saved joint fullstate differs from evaluation snapshot')
    result={};refs={}
    for e in c['evaluations']:
        if e['role']!='validation' or e['backend']!='huggingface_frozen_causal_prefill':raise ValueError('No final pool may enter acceptance')
        bank=a.read(a.checked(e['bank']))[e['split']];frozen=frozen_validation(bank,a.read(a.checked(e['fixed_g5'])))
        ref=receipt['predictions'][e['name']]
        with gzip.open(a.checked(ref),'rt') as f:rows=[json.loads(line) for line in f]
        result[e['name']]=accept_rows(rows,frozen);compare(result[e['name']],receipt['metrics'][e['name']]);refs[e['name']]=ref
    done=dict(status='JOINT_HIDDEN_COUNT_FIXED_IDS_OWN_EM_DIRECT_Q_METRICS_FULL_CP_INDEPENDENTLY_ACCEPTED',cursor=cursor,
        config=config,checkpoint=verified,evaluation_receipt=a.proof(directory/'complete.json'),predictions=refs,metrics=result,
        all_nine_states_unchanged_during_evaluation=True,threshold=.5,threshold_fitted=False,
        actor_updates=0,parameters_fitted=False,final_pool_read=False,raw_predictions_stay_remote=True)
    a.immutable(directory/'independent-audit.json',done);return done


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--cursor',type=int,required=True);args=p.parse_args()
    result=audit(args.base,args.cursor)
    print(json.dumps(dict(status=result['status'],cursor=result['cursor'],metrics=result['metrics'],
        audit=a.proof(Path(args.base)/'evaluations'/('cursor-%06d'%args.cursor)/'independent-audit.json'))))
