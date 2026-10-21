"""Stdlib, read-only independent own-new5 audit for the frozen-E1 screen.

Only metadata, count-head prediction rows and the small frozen-logit cache are
read. No head/model/Adam payload is opened; no prediction/parameter is returned.
The producer's rows omit trajectory IDs: their executed-input binding remains
inherited from the immutable scorer/config. This auditor independently checks
the frozen own-five join and the cache's full ordered trajectory identity SHA.
"""
import argparse
import array
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

EPOCHS = (0, 1, 2, 4, 8, 20)
MODES = ('raw', 'EMA')
CONFIG_SHA = '0be60cc806a177790dce39fa28269303087a1b43ede90a439cce8a63f9e3a0fc'
HALVES_SHA = '3c1c89c44bb90884a7c7ba5f5b9b5c2292f1a4547bd7e25cfa082c24eed0c9e9'


def canonical(v):
    return json.dumps(v, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def digest(v):
    return hashlib.sha256(canonical(v)).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def proof(path):
    p = Path(path)
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return dict(path=str(p), bytes=p.stat().st_size, sha256=h.hexdigest())


def checked(ref):
    if proof(ref['path']) != {k: ref[k] for k in ('path', 'bytes', 'sha256')}:
        raise ValueError('Exact source SHA/bytes changed')
    return Path(ref['path'])


def immutable(path, value):
    p = Path(path)
    raw = canonical(value) + b'\n'
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        if p.read_bytes() != raw:
            raise ValueError('An accepted audit already exists with different content')
    else:
        tmp = p.with_name(p.name + '.tmp')
        tmp.write_bytes(raw)
        tmp.replace(p)


def rank_auc(rows):
    """Independent pair-concordance count, with one half for score ties."""
    positives = sum(0 < r['K'] < 5 for r in rows)
    negatives = len(rows) - positives
    if not positives or not negatives:
        return None
    ordered = sorted(rows, key=lambda r: r['q'])
    wins = 0.0
    lower_negatives = 0
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j]['q'] == ordered[i]['q']:
            j += 1
        p = sum(0 < r['K'] < 5 for r in ordered[i:j])
        n = j - i - p
        wins += p * (lower_negatives + n / 2)
        lower_negatives += n
        i = j
    return wins / (positives * negatives)


def metrics(rows):
    for r in rows:
        if type(r['K']) is not int or not 0 <= r['K'] <= 5 or not math.isfinite(r['q']) or not 0 <= r['q'] <= 1:
            raise ValueError('Finite score/strict own-G5 K required')
    hist = [sum(r['K'] == k for r in rows) for k in range(6)]
    tp = sum(0 < r['K'] < 5 and r['q'] >= .5 for r in rows)
    tn = sum(r['K'] in (0, 5) and r['q'] < .5 for r in rows)
    positive, negative = sum(hist[1:5]), hist[0] + hist[5]
    tpr = tp / positive if positive else None
    tnr = tn / negative if negative else None
    top = sorted(rows, key=lambda r: (-r['q'], r['qid']))[:len(rows) // 4]
    def fpr(k):
        return sum(r['K'] == k and r['q'] >= .5 for r in rows) / hist[k] if hist[k] else None
    return dict(questions=len(rows), K_histogram=hist, Mixed_BA=(tpr + tnr) / 2 if positive and negative else None,
        Mixed_recall=tpr, nonMixed_recall=tnr, allwrong_FPR=fpr(0), allcorrect_FPR=fpr(5),
        Mixed_AUC=rank_auc(rows), predicted_Mixed_fraction=sum(r['q'] >= .5 for r in rows) / len(rows) if rows else None,
        confusion=dict(TP=tp, FN=positive-tp, TN=tn, FP=negative-tn),
        Top25percent=dict(questions=len(top), Mixed_count=sum(0 < r['K'] < 5 for r in top),
            Mixed_fraction=sum(0 < r['K'] < 5 for r in top) / len(top) if top else None,
            K_histogram=[sum(r['K'] == k for r in top) for k in range(6)],
            ordered_qids_sha256=digest([r['qid'] for r in top])))


def expected_halves(nq):
    if len(nq) != 2560 or len(set(nq)) != 2560:
        raise ValueError('Exactly 2560 distinct NQ identities required')
    ordered = sorted(nq, key=lambda q: (digest(['large-count-NQ-halves', 9137, q]), q))
    return [ordered[:1280], ordered[1280:]]


def partitions(rows, halves):
    a, b = map(set, halves)
    return dict(pooled=metrics(rows), NQ=metrics([r for r in rows if r['qid'] in a or r['qid'] in b]),
        HotpotQA=metrics([r for r in rows if r['qid'] not in a and r['qid'] not in b]),
        half0=metrics([r for r in rows if r['qid'] in a]), half1=metrics([r for r in rows if r['qid'] in b]))


def gate(candidate, baseline):
    """Only the preregistered NQ/two-halves gate; no thresholds fitted."""
    whole = candidate['NQ']['Mixed_BA'] is not None and baseline['NQ']['Mixed_BA'] is not None
    whole = whole and candidate['NQ']['Mixed_BA'] > baseline['NQ']['Mixed_BA'] and candidate['NQ']['Top25percent']['Mixed_count'] > baseline['NQ']['Top25percent']['Mixed_count']
    details = dict(NQ_strict_BA_and_Top640=bool(whole))
    for part in ('half0', 'half1'):
        c, b = candidate[part], baseline[part]
        details[part + '_BA_and_Top320_noninferior'] = bool(c['Mixed_BA'] is not None and b['Mixed_BA'] is not None and c['Mixed_BA'] >= b['Mixed_BA'] and c['Top25percent']['Mixed_count'] >= b['Top25percent']['Mixed_count'])
    return dict(pass_internal_node=all(details.values()), conditions=details)


def validate_sources(c, cohort, selection, fixed_rows, all_groups):
    qids, fit = cohort['holdout_qids'], cohort['fit_qids']
    if len(fit) != 30016 or len(set(fit)) != 30016 or len(qids) != 5120 or len(set(qids)) != 5120 or set(fit) & set(qids):
        raise ValueError('Frozen distinct 30016 fit / 5120 holdout identities required')
    fixed = {g['qid']: g for g in fixed_rows}
    selected = {g['qid']: g for g in selection['groups']}
    groups = {g['qid']: g for g in all_groups}
    if len(fixed) != len(fixed_rows) or len(selected) != 5120 or set(fixed) != set(qids) or set(selected) != set(qids):
        raise ValueError('Frozen own-G5 selection coverage differs')
    if any('labels' in r or 'strict_em' in r for r in selection['groups']):
        raise ValueError('Precommitted selection must be label-blind identity metadata')
    domains = dict(NQ=0, HotpotQA=0)
    tids = []
    for q in fit + qids:
        g = groups[q]
        if len(g['trajectory_ids']) != 10 or len(set(g['trajectory_ids'])) != 10 or len(g['labels']) != 10 or any(type(v) is not int or v not in (0, 1) for v in g['labels']):
            raise ValueError('Ten real own trajectory IDs/strict EM required')
        tids.extend(g['trajectory_ids'])
    for q in qids:
        g, f, s = groups[q], fixed[q], selected[q]
        if s['source_indices'] != [5, 6, 7, 8, 9] or s['trajectory_ids'] != g['trajectory_ids'][5:] or f['trajectory_ids'] != s['trajectory_ids'] or f['labels'] != g['labels'][5:] or f['question_hash'] != g['question_hash'] or s['question_hash'] != g['question_hash']:
            raise ValueError('Committed own-new5 IDs/EM/question hash changed')
        name = s['domain']
        if name not in domains or f['domain'] != name:
            raise ValueError('Frozen domain metadata differs')
        domains[name] += 1
    if domains != dict(NQ=2560, HotpotQA=2560) or len(set(tids)) != len(tids):
        raise ValueError('Own trajectory identity/domain counts differ')
    return fixed, tids


def verify_rows(path, qids, fixed):
    rows = []
    seen = set()
    with gzip.open(path, 'rt') as f:
        for line in f:
            r = json.loads(line)
            q = r['qid']
            if q in seen or q not in fixed:
                raise ValueError('Duplicate or foreign holdout qid')
            seen.add(q)
            own = fixed[q]
            if type(r['K']) is not int or r['K'] != sum(own['labels']) or r['domain'] != own['domain']:
                raise ValueError('Prediction K/domain does not match own strict EM')
            if 'trajectory_ids' in r and r['trajectory_ids'] != own['trajectory_ids']:
                raise ValueError('Prediction own IDs differ')
            if not math.isfinite(r['count_NLL']) or r['count_NLL'] < 0 or type(r['predicted_K']) is not int or not 0 <= r['predicted_K'] <= 5:
                raise ValueError('Finite count diagnostics required')
            rows.append(r)
    if len(rows) != len(qids) or seen != set(qids) or [r['qid'] for r in rows] != qids:
        raise ValueError('Exact frozen holdout row order/coverage differs')
    metrics(rows)
    return rows


def compare_producer(rows, actual, declared):
    keys = ('questions', 'K_histogram', 'Mixed_BA', 'Mixed_recall', 'nonMixed_recall', 'Mixed_AUC', 'predicted_Mixed_fraction')
    for k in keys:
        a, b = actual[k], declared[k]
        if isinstance(a, float) and isinstance(b, (float, int)):
            if not math.isclose(a, b, rel_tol=0, abs_tol=1e-12):
                raise ValueError('Producer metric differs: ' + k)
        elif a != b:
            raise ValueError('Producer metric differs: ' + k)
    for k in ('questions', 'Mixed_count', 'Mixed_fraction'):
        a, b = actual['Top25percent'][k], declared['Top25percent'][k]
        if isinstance(a, float) and isinstance(b, (float, int)):
            if not math.isclose(a, b, rel_tol=0, abs_tol=1e-12):
                raise ValueError('Producer Top25 metric differs')
        elif a != b:
            raise ValueError('Producer Top25 metric differs')
    if not math.isclose(sum(r['count_NLL'] for r in rows) / len(rows), declared['count_NLL'], rel_tol=0, abs_tol=1e-10):
        raise ValueError('Producer count NLL differs')
    if not math.isclose(sum(r['K'] == r['predicted_K'] for r in rows) / len(rows), declared['count_accuracy'], rel_tol=0, abs_tol=1e-12):
        raise ValueError('Producer count accuracy differs')


def audit(base, output):
    base, output = Path(base), Path(output)
    config_ref = proof(base / 'config.json')
    if config_ref['sha256'] != CONFIG_SHA:
        raise ValueError('Only the predeclared immutable large-screen config is accepted')
    c = read(config_ref['path'])
    if c['evaluation_epochs'] != list(EPOCHS) or c['external_pool_access'] or c['final_pool_access']:
        raise ValueError('Frozen screen scope differs')
    half_ref = proof(base / 'screen-protocol/frozen-NQ-halves.json')
    if half_ref['sha256'] != HALVES_SHA:
        raise ValueError('Pre-score label-blind NQ halves changed')
    frozen = read(half_ref['path'])
    cohort = read(checked(c['cohort']))
    source_refs = {k: c['inputs'][k] for k in ('internal_holdout_selection', 'internal_holdout_groups', 'merged_groups', 'identity_exclusion_acceptance')}
    values = {k: read(checked(ref)) for k, ref in source_refs.items()}
    exclusion = values['identity_exclusion_acceptance']
    if exclusion.get('labels_consulted_for_selection') is not False or exclusion.get('scores_consulted_for_selection') is not False or any(v != 0 for v in exclusion['holdout_fixed_evaluation_qid_and_hash_overlap'].values()) or any(v != 0 for v in exclusion['merged_fixed_evaluation_qid_and_hash_overlap'].values()):
        raise ValueError('Identity-only holdout exclusion proof differs')
    fixed, tids = validate_sources(c, cohort, values['internal_holdout_selection'], values['internal_holdout_groups']['internal_holdout'], values['merged_groups']['train'])
    qids = cohort['holdout_qids']
    halves = expected_halves([q for q in qids if fixed[q]['domain'] == 'NQ'])
    if frozen['config'] != config_ref or frozen['selection'] != source_refs['internal_holdout_selection'] or frozen['halves'] != halves or frozen['half_SHA256'] != [digest(h) for h in halves] or frozen['labels_read'] or frozen['scores_read']:
        raise ValueError('Frozen half identity reconstruction differs')
    cache_path = base / 'cache/complete.json'
    if not cache_path.exists():
        return dict(status='WAITING_FOR_ACTUAL_FROZEN_CACHE_COMPLETE',config=config_ref,actual_scored_nodes=0,external_scoring_permitted=False)
    cache_ref, cache = proof(cache_path), read(cache_path)
    if cache['config_sha256'] != CONFIG_SHA or cache['trajectory_ids_sha256'] != digest(tids) or cache['frozen_legacy_before'] != cache['frozen_legacy_after'] or cache['actor_updates'] != 0 or cache['legacy_updates'] != 0:
        raise ValueError('Frozen small-logit cache lineage differs')
    if sys.byteorder != 'little':
        raise ValueError('Declared native f32 cache requires little endian')
    z = array.array('f')
    z.frombytes(checked(cache['files']['logits']).read_bytes())
    if len(z) != len(tids):
        raise ValueError('Frozen small-logit cache count differs')
    baseline_rows = []
    for i, q in enumerate(qids):
        vals = z[(len(cohort['fit_qids']) + i) * 10 + 5:(len(cohort['fit_qids']) + i + 1) * 10]
        if len(vals) != 5 or not all(math.isfinite(v) for v in vals):
            raise ValueError('Finite own-new5 frozen logits required')
        ps = [1 / (1 + math.exp(-v)) if v >= 0 else math.exp(v) / (1 + math.exp(v)) for v in vals]
        product_correct, product_wrong = 1., 1.
        for p in ps:
            product_correct *= p
            product_wrong *= 1 - p
        baseline_rows.append(dict(qid=q,q=min(1.,max(0.,1-product_correct-product_wrong)),K=sum(fixed[q]['labels'])))
    baseline = partitions(baseline_rows, halves)
    common = dict(config=config_ref,source_refs=source_refs,cohort=c['cohort'],frozen_halves=half_ref,cache_complete=cache_ref,
        frozen_small_logits=cache['files']['logits'],own_new5_IDs_sha256=digest([fixed[q]['trajectory_ids'] for q in qids]),
        own_new5_strict_EM_sha256=digest([fixed[q]['labels'] for q in qids]),ordered_holdout_qids_sha256=digest(qids),
        baseline_arithmetic='float64 stable sigmoid of stored native float32 Z; float64 product; fixed0.5',
        per_prediction_GPU_forward_IDs_independently_observed=False,
        forward_ID_binding='Inherited immutable scorer/config/cache; own5 metadata join independently verified',
        model_or_head_or_optimizer_payload_opened=False,external_payload_opened=False,final_payload_opened=False,
        parameters_or_thresholds_fitted=False)
    baseline_file = output / 'frozen-E1-baseline.json'
    immutable(baseline_file,dict(status='INDEPENDENT_SAME_CACHE_FROZEN_E1_PRODUCT_BASELINE_PASSED',metrics=baseline,**common))
    nodes=[]
    for epoch in EPOCHS:
        done_path=base/'evaluations'/('epoch-%02d'%epoch)/'complete.json'
        if not done_path.exists():
            continue
        done_ref, done=proof(done_path),read(done_path)
        cp=done['checkpoint']
        if done['epoch'] != epoch or done['cursor'] != epoch*938 or not done['training_components_unchanged'] or done['external_pool_access'] or done['final_pool_access'] or cp['cursor'] != epoch*938 or cp['new_head_Adam_steps'] != epoch*938 or not cp['full_CPU_reload_passed']:
            raise ValueError('Completed epoch/checkpoint metadata is not accepted')
        for mode in MODES:
            path=output/('epoch-%02d-%s.json'%(epoch,mode))
            if path.exists():
                accepted=read(path)
                if any(accepted[k]!=v for k,v in common.items()) or accepted['evaluation_complete']!=done_ref or accepted['prediction']!=done['predictions'][mode+'-internal_holdout_new5'] or accepted['checkpoint_receipt']!=cp:
                    raise ValueError('An accepted completed-node source changed')
                nodes.append(dict(epoch=epoch,model=mode,metrics=accepted['metrics'],screen_gate=accepted['screen_gate'],audit=proof(path)))
                continue
            ref=done['predictions'][mode+'-internal_holdout_new5']
            rows=verify_rows(checked(ref),qids,fixed)
            actual=partitions(rows,halves)
            declared=done['metrics'][mode]['internal_holdout_new5']
            compare_producer(rows,actual['pooled'],declared)
            for name in ('NQ','HotpotQA'):
                domain_rows=[r for r in rows if r['domain']==name]
                compare_producer(domain_rows,actual[name],declared['by_domain'][name])
            value=dict(status='INDEPENDENT_NODE_OWN_NEW5_AND_METRICS_PASSED',epoch=epoch,model=mode,
                metrics=actual,screen_gate=gate(actual,baseline),prediction=ref,evaluation_complete=done_ref,
                checkpoint_receipt=cp,checkpoint_payload_rehashed_or_loaded=False,**common)
            immutable(path,value)
            nodes.append(dict(epoch=epoch,model=mode,metrics=actual,screen_gate=value['screen_gate'],audit=proof(path)))
    all_nodes=len(nodes)==12
    selected=None
    if all_nodes:
        eligible=[n for n in nodes if n['screen_gate']['pass_internal_node']]
        eligible.sort(key=lambda n:(-n['metrics']['NQ']['Top25percent']['Mixed_count'],-n['metrics']['NQ']['Mixed_BA'],n['epoch'],0 if n['model']=='EMA' else 1))
        if eligible:selected=dict(epoch=eligible[0]['epoch'],model=eligible[0]['model'],audit=eligible[0]['audit'])
    summary=dict(status='ALL12_NODES_INDEPENDENTLY_AUDITED' if all_nodes else 'AVAILABLE_NODES_INDEPENDENTLY_AUDITED_SCREEN_INCOMPLETE',
        baseline=proof(baseline_file),baseline_metrics=baseline,actual_scored_nodes=len(nodes),nodes=nodes,
        all12_required_before_internal_selection=True,selected_internal_candidate=selected,
        external_scoring_permitted=False,external_requires_separate_online_micro8_qualification=True,**common)
    # Each snapshot is content-addressed; already accepted node proofs stay immutable.
    target=output/('snapshot-'+digest([n['audit'] for n in nodes])[:16]+'.json')
    immutable(target,summary)
    return dict(summary,receipt=proof(target))


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    print(json.dumps(audit(a.base,a.output),sort_keys=True),flush=True)


if __name__=='__main__':
    main()
