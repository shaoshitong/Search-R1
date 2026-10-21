"""Training-only descriptive consistency of two authentic independent G5 draws.

No model fit, evaluation/final labels, tensors, or new rollouts. This is not a
Bayes ceiling: each draw has its own causal prefix and its own strict-EM label.
"""
from collections import defaultdict
import hashlib, json, math, os, resource, time
from pathlib import Path
from reproduction.operations.qwen7b_warmup import train_g10_domain_statistics as d

OUT = d.A / 'training-G10-independent-seed-consistency-v1'


def summarize(rows):
    joint = [[0] * 6 for _ in range(6)]
    for g in rows:
        joint[sum(g['labels'][:5])][sum(g['labels'][5:])] += 1
    n = len(rows)
    old = [sum(r) for r in joint]
    new = [sum(r[k] for r in joint) for k in range(6)]
    # Rows/columns: nonMixed, Mixed. True label always belongs to its own G5.
    mixed_joint = [[0, 0], [0, 0]]
    for i in range(6):
        for j in range(6):
            mixed_joint[int(0 < i < 5)][int(0 < j < 5)] += joint[i][j]
    tn, fp = mixed_joint[0]
    fn, tp = mixed_joint[1]
    den = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    assert sum(map(sum, joint)) == n == sum(old) == sum(new)
    return dict(questions=n, authentic_G5_draws=2*n, authentic_trajectories=10*n,
                old5_K0to5=old, new5_K0to5=new, joint_Kold_rows_Knew_columns=joint,
                old_new_Mixed_2x2=mixed_joint,
                old_actual_Mixed_fraction=sum(old[1:5])/n,
                new_actual_Mixed_fraction=sum(new[1:5])/n,
                own_label_Mixed_agreement=(tp+tn)/n,
                own_label_Mixed_disagreement=(fp+fn)/n,
                old_Mixed_and_new_Mixed=tp,
                new_Mixed_given_old_Mixed=tp/(tp+fn) if tp+fn else None,
                old_Mixed_given_new_Mixed=tp/(tp+fp) if tp+fp else None,
                old_new_Mixed_MCC=(tp*tn-fp*fn)/den if den else None,
                old5_Mixed_as_predictor_of_new5_BA=.5*(tp/(tp+fp)+tn/(tn+fn))
                if (tp+fp)*(tn+fn) else None,
                descriptive_two_draws_not_model_metrics_or_noise_ceiling=True)


def main():
    started=time.time()
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    resource.setrlimit(resource.RLIMIT_AS, (3*1024**3, 3*1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
    OUT.mkdir(parents=True, exist_ok=True)
    dest=OUT/'complete.json'
    if dest.exists():
        print(json.dumps(dict(receipt=d.proof(dest), result=d.read(dest))));return
    accepted=[
        ('initial31232', d.A/'peer-newseed-ingest-newseed-20261021-185747-445fc691/complete.json',
         '1fd5c29862267bc202419973ba6238d9c7b66de7ef2199dba725dda7fecf0804',31232),
        ('new15616', d.A/'peer-newseed-ingest-audited122-delta-20261021-673b8cb6/complete.json',
         '13f7e6a24699dd9858387210552a4d922ab521bd9a9c7c99648dab147f36048e',15616)]
    cohorts={};refs={};allrows=[];seen=set();seenhash=set()
    for name,path,sha,n in accepted:
        done=d.read(d.pinned(path,sha))
        assert done['status']=='RECEIVED_NEWSEED5_PLUS_CANONICAL_OLD5_RAW_G10_STRICT_EM_CAUSAL_SEEDS_ACCEPTED'
        assert done['strict_EM_mismatches']==0 and done['actual_same_question_old_new_seed_collisions']==0
        rows=d.groups(d.checked(done['outputs']['groups']),n)
        for g in rows:
            assert g['qid'] not in seen and g['question_hash'] not in seenhash
            seen.add(g['qid']);seenhash.add(g['question_hash'])
        assert d.summarize(rows)['K0_to10']==done['histogram_K0_to10']
        cohorts[name]=summarize(rows);refs[name]=dict(accepted_complete=d.proof(path),groups=done['outputs']['groups']);allrows+=rows
    domains=defaultdict(list)
    for g in allrows:domains[json.loads(g['qid'])[0]].append(g)
    result=dict(status='AUTHENTIC_TRAINING_G5_SEED_CONSISTENCY_COMPLETE',source_refs=refs,
                cohorts=cohorts,all=summarize(allrows),domains={k:summarize(v)for k,v in sorted(domains.items())},
                label_definition='0 < sum(own five strict EM) < 5; never reuse other draw labels',
                interpretation='Descriptive association between two seeds. Not a learned selector, calibration fit, model evaluation, or Bayes/noise ceiling; per-draw prefixes differ.',
                external_or_final_labels_read=False,feature_or_model_tensor_read=False,
                GPU_used=False,training_or_plan_changed=False,
                seconds=time.time()-started,epoch=time.time())
    temp=dest.with_suffix('.tmp');temp.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n');temp.replace(dest)
    print(json.dumps(dict(receipt=d.proof(dest),result=result)))


if __name__=='__main__':main()
