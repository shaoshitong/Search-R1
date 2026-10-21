"""Metadata-only exact-input ambiguity strata for the fixed2048 ownG5 pool.

No tensor/checkpoint/raw rollout load or new model scoring. Exact feature
identity relies on accepted reconstructed-row SHA plus masks/positions, never
on token identity alone. Output contains aggregates, not question/probability
rows. New checkpoint predictions may be appended without changing strata.
"""
import argparse,collections,gzip,hashlib,json,math,time
from pathlib import Path
FIXED_G5='c55de2b4f66f45dcd94a8e246bfca97e53bdedc4ca85180c75e60c574dfc04b7'
ROW_ACCEPTANCE='ALL_10240_NEW_FIXED_G5_HF_ROWS_RECONSTRUCTED_RETAINED_SHA_PASSED'


class UnsupportedFingerprint(ValueError):pass


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def proof(path):
    p=Path(path);h=hashlib.sha256()
    with p.open('rb')as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())


def checked(ref):
    p=Path(ref['path'])
    if not p.is_absolute():raise ValueError('Absolute source reference required')
    actual=proof(p)
    if actual['sha256']!=ref['sha256']or('bytes'in ref and actual['bytes']!=ref['bytes']):raise ValueError('Small source metadata/prediction SHA or bytes differs: '+str(p))
    return p


def read(ref):return json.loads(checked(ref).read_text())


def own_validation(complete):
    source=read(complete['source_groups'])['validation'];plan=read(complete['source_fixed_G5']);plan=plan.get('fixed_G5',plan)
    if plan['summary'].get('labels_consulted')is not False or plan['summary'].get('role')!='validation'or digest(plan['groups'])!=plan['summary']['selected_g5_sha256']:raise ValueError('Precommitted label-blind ownG5 plan differs')
    by={g['qid']:g for g in source};out={};seen=set()
    if len(by)!=len(source)or len(by)!=2048 or len(plan['groups'])!=2048:raise ValueError('Exact fixed2048 question coverage required')
    for g in plan['groups']:
        qid=g['qid'];tids=g['trajectory_ids'];original=by[qid];ys=dict(zip(original['trajectory_ids'],original['labels']))
        if qid in out or len(tids)!=5 or len(set(tids))!=5 or not set(tids)<=set(ys)or seen.intersection(tids):raise ValueError('Repeated/non-own fixedG5 IDs')
        labels=[ys[t]for t in tids]
        if any(type(y)is not int or y not in(0,1)for y in labels):raise ValueError('Authentic binary own EM required')
        seen.update(tids);out[qid]=dict(qid=qid,trajectory_ids=tids,strict_em=labels)
    return out


def input_keys(row,manifest):
    """Zero padding and Q positions are deterministic from the retained lengths."""
    qlen=row['question_length'];plen=row['policy_length'];positions=tuple(row['policy_positions'])
    if type(qlen)is not int or not 1<=qlen<=273 or type(plen)is not int or not 0<=plen<=20 or len(positions)!=plen or any(type(p)is not int or p<0 for p in positions):raise UnsupportedFingerprint('Index lacks exact mask/causal-position geometry')
    for key in('policy_token_sha256','retained_feature_sha256'):
        value=row.get(key)
        if not isinstance(value,str)or len(value)!=64 or any(c not in'0123456789abcdef'for c in value):raise UnsupportedFingerprint('Sealed index has no strict '+key)
    token=(row['policy_token_sha256'],positions)
    strict=(row['retained_feature_sha256'],qlen,plen,positions,manifest['capture_layer'],manifest['dimension'],manifest['question_max_tokens'],manifest['prefix_tokens'],manifest['model_revision'],manifest.get('feature_dtype','bfloat16_bits_uint16'))
    return token,strict


def grouping(keys,labels):
    counts=collections.Counter(keys);pairs=sum(n*(n-1)//2 for n in counts.values())
    conflict=sum(keys[i]==keys[j]and labels[i]!=labels[j]for i in range(5)for j in range(i+1,5))
    return dict(unique_inputs=len(counts),duplicate_trajectories=5-len(counts),duplicate_pairs=pairs,opposite_EM_identical_key_pairs=conflict,has_opposite_EM_identical_key_pair=bool(conflict))


def fingerprint_groups(spec,expected):
    complete=read(spec['validation_complete']);audit=read(spec['validation_reader_audit'])
    if(complete.get('status')!='NEW_VALIDATION2048_FIXED_G5_QUALIFIED_HF_REPLAY_LAYER26_COMPLETE'or complete.get('questions')!=2048 or complete.get('trajectories')!=10240 or complete.get('source_runtime')!='huggingface_frozen_causal_prefill'or complete.get('capture_layers')!=[26]or complete.get('fixed_G5_sha256')!=FIXED_G5):raise UnsupportedFingerprint('Current fixed2048 is not the supported accepted frozen-HF L26 index format')
    if(audit.get('status')!=ROW_ACCEPTANCE or audit.get('complete')!=spec['validation_complete']or audit.get('features')!=complete['features']or audit.get('all_manifest_and_payload_SHA256')is not True or audit.get('all_reconstructed_row_SHA256')is not True):raise ValueError('Prior full reconstructed-row acceptance is required for metadata identity claims')
    actual_own=own_validation(complete)
    if actual_own!=expected:raise ValueError('Prediction ownG5/strictEM differs from actual precommitted fixed pool')
    manifests=read(complete['features'])
    if manifests!=complete['feature_manifests_by_capture_layer']['26']:raise ValueError('Current fixed feature descriptor list changed')
    wanted={tid:q for q,g in expected.items()for tid in g['trajectory_ids']};rows={};refs=[]
    for descriptor in manifests:
        path=checked(descriptor);m=json.loads(path.read_text())
        if m.get('status')!='LOSSLESS_HF_REPLAY_FEATURES_COMPLETE'or m.get('source_runtime')!='huggingface_frozen_causal_prefill'or m.get('model_revision')!='a09a35458c702b33eeacc393d103063234e8bc28'or m.get('feature_dtype','bfloat16_bits_uint16')!='bfloat16_bits_uint16'or m.get('layout')!='lossless_question_base_xor_policy_v1'or m.get('capture_layer')!=26 or m.get('feature_layer')!=26 or m.get('question_max_tokens')!=273 or m.get('prefix_tokens')!=20 or m.get('dimension')!=3584:raise UnsupportedFingerprint('Unsupported native/other replay manifest; no strict feature fingerprint fabricated')
        index=m['files']['index.jsonl'];p=checked(index)
        if p.parent!=path.parent:raise ValueError('Sealed index escaped manifest directory')
        count=0
        with p.open()as stream:
            for line in stream:
                row=json.loads(line);count+=1;tid=row['trajectory_id']
                if tid not in wanted:continue
                if row.get('capture_layer')!=26 or row.get('dimension')!=3584 or row.get('capture_kind')!=m['capture_kind']:raise ValueError('Index layer/geometry differs from sealed manifest')
                if tid in rows or row['qid']!=wanted[tid]:raise ValueError('Repeated/different own replay trajectory binding')
                rows[tid]=input_keys(row,m)
        if count!=m['count']:raise ValueError('Sealed index declared row count differs')
        refs.append(dict(manifest=descriptor,index=index,count=count))
    if set(rows)!=set(wanted):raise ValueError('Exact10240 own retained-input fingerprints not found')
    fingerprints={}
    for q,g in expected.items():
        own=[rows[t]for t in g['trajectory_ids']];fingerprints[q]=dict(tokens_positions=grouping([v[0]for v in own],g['strict_em']),strict_features_masks_positions=grouping([v[1]for v in own],g['strict_em']))
    return fingerprints,dict(validation_complete=spec['validation_complete'],prior_full_reader_acceptance=spec['validation_reader_audit'],manifest_indexes=refs,index_rows=len(rows),question_groups=len(fingerprints),prior_feature_payload_SHA_and_reconstructed_row_proof_inherited=True,tensor_or_payload_loaded_now=False)


def auc(rows):
    values=sorted((r['predicted_mixed'],0<sum(r['strict_em'])<5)for r in rows);pos=sum(y for _,y in values);neg=len(values)-pos
    if not pos or not neg:return None
    total=0.;i=0
    while i<len(values):
        j=i+1
        while j<len(values)and values[j][0]==values[i][0]:j+=1
        total+=sum(y for _,y in values[i:j])*(i+1+j)/2;i=j
    return (total-pos*(pos+1)/2)/(pos*neg)


def metrics(rows,global_top):
    classes={name:dict(questions=0,predicted_Mixed=0)for name in('allwrong','Mixed','allcorrect')};hist=[0]*6;selected_hist=[0]*6
    for r in rows:
        k=sum(r['strict_em']);name='allwrong'if k==0 else'allcorrect'if k==5 else'Mixed';classes[name]['questions']+=1;classes[name]['predicted_Mixed']+=r['predicted_mixed']>=.5;hist[k]+=1
        if r['qid']in global_top:selected_hist[k]+=1
    for name,v in classes.items():
        n=v['questions'];error=n-v['predicted_Mixed']if name=='Mixed'else v['predicted_Mixed'];v.update(error_count=error,error_rate=error/n if n else None)
    positive=classes['Mixed']['questions'];negative=len(rows)-positive;tp=classes['Mixed']['predicted_Mixed'];fp=classes['allwrong']['predicted_Mixed']+classes['allcorrect']['predicted_Mixed'];tpr=tp/positive if positive else None;tnr=1-fp/negative if negative else None;selected=sum(selected_hist)
    return dict(questions=len(rows),trueMixed=positive,trueMixed_fraction=positive/len(rows)if rows else None,K_histogram=hist,MixedBA=(tpr+tnr)/2 if tpr is not None and tnr is not None else None,Mixed_TPR=tpr,nonMixed_TNR=tnr,AUC=auc(rows),three_true_class_errors=classes,globalTop512=dict(selected_questions=selected,actualMixed=sum(selected_hist[1:5]),actualMixed_fraction=sum(selected_hist[1:5])/selected if selected else None,K_histogram=selected_hist,stratum_coverage=selected/len(rows)if rows else None),per_stratum_reselection_performed=False)


def audit_score(rows,fingerprints):
    top={r['qid']for r in sorted(rows,key=lambda r:(-r['predicted_mixed'],str(r['qid'])))[:512]};out=dict(global_pool=metrics(rows,top),strata={})
    for field in('tokens_positions','strict_features_masks_positions'):
        out['strata'][field]={}
        for unique in range(1,6):
            group=[r for r in rows if fingerprints[r['qid']][field]['unique_inputs']==unique]
            out['strata'][field]['unique%d'%unique]=dict(metrics(group,top),duplicate_trajectories_per_question=5-unique,duplicate_pairs=sum(fingerprints[r['qid']][field]['duplicate_pairs']for r in group),opposite_EM_identical_key_pairs=sum(fingerprints[r['qid']][field]['opposite_EM_identical_key_pairs']for r in group),questions_with_opposite_EM_identical_key_pair=sum(fingerprints[r['qid']][field]['has_opposite_EM_identical_key_pair']for r in group))
    out['cross_strata']={}
    for token_unique in range(1,6):
        for strict_unique in range(1,6):
            group=[r for r in rows if fingerprints[r['qid']]['tokens_positions']['unique_inputs']==token_unique and fingerprints[r['qid']]['strict_features_masks_positions']['unique_inputs']==strict_unique]
            if group:out['cross_strata']['tokens%d_strict%d'%(token_unique,strict_unique)]=metrics(group,top)
    return out


def run(packet,output):
    started=time.perf_counter();packet_path=Path(packet);p=json.loads(packet_path.read_text());c=read(p['source_config']);expected=None;predictions={}
    accepted=[read(ref)for ref in p['accepted_evaluation_completes']]
    def contains(value,ref):
        if isinstance(value,dict):return value==ref or any(contains(v,ref)for v in value.values())
        if isinstance(value,list):return any(contains(v,ref)for v in value)
        return False
    for item in p['predictions']:
        name=item['name'];path=checked(item['file'])
        if not any(contains(receipt,item['file'])for receipt in accepted):raise ValueError('Prediction descriptor is not in a SHA-accepted evaluation receipt')
        if name in predictions:raise ValueError('Repeated model name')
        with gzip.open(path,'rt')as stream:rows=[json.loads(line)for line in stream]
        own={r['qid']:dict(qid=r['qid'],trajectory_ids=r['trajectory_ids'],strict_em=r['strict_em'])for r in rows}
        if len(rows)!=2048 or len(own)!=2048:raise ValueError('Only complete fixed2048 predictions')
        for r in rows:
            if r.get('true_mixed') is not None and r['true_mixed']!=(0<sum(r['strict_em'])<5):raise ValueError('Prediction true-Mixed differs from own EM')
            if len(r['trajectory_ids'])!=5 or len(set(r['trajectory_ids']))!=5 or len(r['strict_em'])!=5 or any(type(y)is not int or y not in(0,1)for y in r['strict_em'])or not math.isfinite(r['predicted_mixed'])or not 0<=r['predicted_mixed']<=1 or 'p_j'in r:raise ValueError('Only actual direct Mixed ownG5 prediction schema')
        if expected is None:expected=own
        elif own!=expected:raise ValueError('Model scores use different own fixed G5/strict EM')
        predictions[name]=rows
    if not predictions:raise ValueError('At least one fully accepted existing model score required')
    try:
        fingerprints,source=fingerprint_groups(c['evaluation'],expected)
    except UnsupportedFingerprint as exc:
        result=dict(status='UNSUPPORTED_FIXED2048_STRICT_INPUT_FINGERPRINT_FORMAT',reason=str(exc),packet=proof(packet_path),no_tensor_or_raw_rollout_loaded=True,GPU_used=False)
    else:
        result=dict(status='FIXED2048_EXISTING_DIRECT_SCORES_EXACT_INPUT_AMBIGUITY_STRATA_COMPLETE',source_config=p['source_config'],packet=proof(packet_path),prediction_sources=p['predictions'],input_source=source,models={name:audit_score(rows,fingerprints)for name,rows in predictions.items()},threshold=.5,globalTop512_order='descending direct probability then string qid; labels never in ordering',strata_label_usage='Fingerprints frozen from input metadata; own EM only for outcomes/error/conflict counts',token_identity_is_feature_identity=False,strict_identity_definition='same accepted retained BF16 feature SHA, lengths/mask, causal positions, geometry/layer/revision/dtype',strict_nonidentity_does_not_measure_numeric_distance=True,group_classifier_irreducible_error_bound_claim=False,no_tensor_checkpoint_or_raw_rollout_loaded=True,full_training_bank_scanned=False,new_model_forward=False,fitted_parameters_or_thresholds=False,final4096_outcomes_read=False,GPU_used=False,elapsed_seconds=time.perf_counter()-started,epoch=time.time())
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():raise ValueError('Existing frozen audit must be read, not overwritten')
    with output.open('x')as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    return dict(receipt=proof(output),**result)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--packet',required=True);parser.add_argument('--output',required=True);a=parser.parse_args();print(json.dumps(run(a.packet,a.output)))
