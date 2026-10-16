"""CPU-only posthoc fixed-G5 mixed diagnostic; never import or change training.

Run remotely: python -m reproduction.prefix20_online.mixed_diagnostic
  --base REMOTE_RUN --out NEW_REMOTE_DIRECTORY [--through-epoch N]
The product formula assumes conditional independence of trajectory correctness.
"""
import argparse
import csv
import datetime
import hashlib
import io
import json
import math
from collections import defaultdict
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def stable_read(path):
    with path.open('rb') as stream:
        import os
        before = os.fstat(stream.fileno())
        data = stream.read()
        after = os.fstat(stream.fileno())
    fields = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if fields(before) != fields(after) or fields(after) != fields(path.stat()):
        raise ValueError('Source changed during read: ' + str(path))
    return data, dict(path=str(path), sha256=digest(data), bytes=len(data),
                      device=before.st_dev, inode=before.st_ino, stat_stable=True)


def read_json(path):
    data, proof = stable_read(path)
    return json.loads(data), proof


def sigmoid_pair(z):
    """Return float64 p and 1-p separately, avoiding cancellation at large logits."""
    z = float(z)
    if not math.isfinite(z):
        raise ValueError('Nonfinite diagnostic logit')
    e = math.exp(-abs(z))
    return (1 / (1 + e), e / (1 + e)) if z >= 0 else (e / (1 + e), 1 / (1 + e))


def mixed_probability(logits):
    """Stable recurrence equal to 1-prod(p)-prod(1-p); no threshold tuning."""
    pairs = [sigmoid_pair(z) for z in logits]
    if len(pairs) != 5:
        raise ValueError('Require complete G5')
    all_one, all_zero = pairs[0]
    mixed = 0.0
    for p, complement in pairs[1:]:
        mixed = math.fsum((mixed, all_zero * p, all_one * complement))
        all_one *= p
        all_zero *= complement
    if not 0.0 <= mixed <= 1.0:
        raise ValueError('Float64 mixed probability outside [0,1]; no clipping applied')
    return mixed, [p for p, _ in pairs]


def rank_auc(scores, labels):
    pos, neg = sum(labels), len(labels) - sum(labels)
    if not pos or not neg:
        return None
    ordered = sorted(zip(scores, labels))
    rank_sum, start = 0.0, 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][0] == ordered[start][0]:
            end += 1
        rank_sum += (start + 1 + end) / 2 * sum(y for _, y in ordered[start:end])
        start = end
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def group_predictions(predictions, index):
    if len(predictions) != len(index) or len({r[0] for r in predictions}) != len(index):
        raise ValueError('Duplicate or incomplete prediction IDs')
    if {r[0] for r in predictions} != set(index):
        raise ValueError('Prediction IDs differ from fixed test IDs')
    groups = defaultdict(list)
    for tid, logit, label in predictions:
        row = index[tid]
        if type(label) is not int or label != row['strict_em']:
            raise ValueError('Prediction label differs from fixed test index')
        sigmoid_pair(logit)
        groups[str(row['qid'])].append((str(tid), float(logit), label, row['question_hash']))
    if len(groups) != 512:
        raise ValueError('Require exactly 512 fixed question groups')
    result = []
    for qid, rows in sorted(groups.items()):
        rows.sort(key=lambda row: row[0])
        if len(rows) != 5 or len({r[3] for r in rows}) != 1:
            raise ValueError('Incomplete G5 or conflicting question identity')
        q, ps = mixed_probability([r[1] for r in rows])
        k = sum(r[2] for r in rows)
        result.append(dict(qid=qid, question_hash=rows[0][3], trajectory_ids=[r[0] for r in rows],
                           probabilities=ps, strict_em=[r[2] for r in rows], correct_count=k,
                           mixed_label=int(0 < k < 5), mixed_probability=q,
                           predicted_mixed=int(q >= 0.5)))
    return result


def aggregate(rows):
    ys = [r['mixed_label'] for r in rows]; qs = [r['mixed_probability'] for r in rows]
    tp = sum(y and q >= .5 for y, q in zip(ys, qs)); fp = sum(not y and q >= .5 for y, q in zip(ys, qs))
    fn = sum(y and q < .5 for y, q in zip(ys, qs)); tn = len(ys) - tp - fp - fn
    pos, neg = sum(ys), len(ys) - sum(ys)
    div = lambda a, b: a / b if b else None
    recall, specificity = div(tp, pos), div(tn, neg)
    # Exact endpoint scoring: impossible observed labels have infinite BCE.
    # JSON uses null plus an explicit infinity flag, without a probability floor.
    wrong_endpoints = sum((y == 1 and q == 0.0) or (y == 0 and q == 1.0)
                          for y, q in zip(ys, qs))
    bce = None if wrong_endpoints else math.fsum(
        -math.log(q) if y else -math.log1p(-q) for y, q in zip(ys, qs)) / len(ys)
    return dict(count=len(ys), mixed_count=pos, nonmixed_count=neg, mixed_fraction=pos/len(ys),
                threshold=.5, tp=tp, fp=fp, fn=fn, tn=tn, accuracy=(tp+tn)/len(ys),
                majority_accuracy=max(pos, neg)/len(ys), balanced_accuracy=(recall+specificity)/2 if pos and neg else None,
                precision=div(tp, tp+fp), recall=recall, specificity=specificity, f1=div(2*tp, 2*tp+fp+fn),
                mean_q=math.fsum(qs)/len(qs), bce=bce,
                bce_is_positive_infinity=bool(wrong_endpoints), wrong_exact_endpoint_count=wrong_endpoints,
                brier=sum((q-y)**2 for q,y in zip(qs,ys))/len(ys), auroc=rank_auc(qs,ys),
                selected_q_ge_0_5_count=tp+fp, actual_mixed_among_selected=div(tp,tp+fp))


def run(base, out, through=None):
    base, out = Path(base).resolve(), Path(out).resolve()
    if not str(base).startswith('/mnt/') or not str(out).startswith('/mnt/'):
        raise ValueError('Run and full per-question output must remain remote /mnt storage')
    if out == base or base in out.parents and not str(out.relative_to(base)).startswith('preparation/'):
        raise ValueError('Output must be separate or under base/preparation')
    cfg, cfg_proof = read_json(base/'preparation/launch-config.json')
    if cfg.get('base') != str(base):
        raise ValueError('Launch config base identity differs')
    latest, latest_proof = read_json(base/'train/latest.json')
    maximum = latest['epoch']
    through = maximum if through is None else through
    if type(through) is not int or not 1 <= through <= maximum:
        raise ValueError('Requested range exceeds the latest completed snapshot')
    cache = Path(cfg['cached_features']); manifest, manifest_proof = read_json(cache/'manifest.json')
    if manifest_proof['sha256'] != cfg['cached_features_manifest_sha256']:
        raise ValueError('Fixed diagnostic manifest binding differs')
    raw, index_proof = stable_read(cache/'index.jsonl')
    if index_proof['sha256'] != manifest['files']['index.jsonl']['sha256']:
        raise ValueError('Fixed identity index SHA differs')
    index = {}
    for line in raw.splitlines():
        row = json.loads(line)
        if row['split'] != 'test': continue
        tid = row['trajectory_id']
        if tid in index or type(row['strict_em']) is not int or row['strict_em'] not in (0,1):
            raise ValueError('Invalid fixed test identity')
        index[tid] = row
    if len(index) != 2560: raise ValueError('Require fixed 2560 trajectories')
    out.mkdir(parents=True, exist_ok=False)
    report = dict(role='POSTHOC_FIXED512_G5_MIXED_PRODUCT_DIAGNOSTIC', run_id=cfg['run_id'], base=str(base),
                  through_epoch=through, latest_complete_at_snapshot=maximum, fixed_manifest_sha256=manifest_proof['sha256'],
                  threshold=.5, formula='1-prod(p_j)-prod(1-p_j)', probability_assumption='conditional independence; not a separately calibrated mixed classifier',
                  model_loaded=False, gpu_used=False, training_modified=False, provenance=dict(config=cfg_proof, latest=latest_proof, manifest=manifest_proof, index=index_proof),
                  epochs=[], issues=[])
    true_identity = None
    import torch  # CPU deserialization of retained prediction tuples only; no model state.
    for epoch in range(1, through+1):
        root = base/'train/epochs'/f'epoch-{epoch:04d}'
        paths = [root/n for n in ('complete.json','metrics.json','diagnostic-predictions.pt')]
        missing = [str(p) for p in paths if not p.is_file()]
        if missing:
            report['issues'].append(dict(epoch=epoch, status='MISSING', paths=missing)); continue
        try:
            complete, cproof = read_json(paths[0]); metrics, mproof = read_json(paths[1])
            if complete['epoch'] != epoch or metrics['epoch'] != epoch or complete['status'] != 'FRESH_ONLINE20_EPOCH_COMPLETE':
                raise ValueError('Epoch is not completed')
            if metrics['test_manifest_sha256'] != manifest_proof['sha256']:
                raise ValueError('Epoch diagnostic binding differs')
            data, pproof = stable_read(paths[2]); saved = torch.load(io.BytesIO(data), map_location='cpu', weights_only=False)
            if saved['epoch'] != epoch or saved['binding_sha256'] != complete['binding_sha256']:
                raise ValueError('Prediction binding differs from that epoch completion')
            rows = group_predictions(saved['predictions'], index)
            identity = digest(json.dumps([(r['qid'],r['trajectory_ids'],r['strict_em']) for r in rows],sort_keys=True).encode())
            if true_identity is not None and identity != true_identity: raise ValueError('True G5 groups changed across epochs')
            true_identity = identity
            with (out/f'epoch-{epoch:04d}-questions.jsonl').open('x') as f:
                for row in rows: f.write(json.dumps(dict(epoch=epoch,**row),allow_nan=False)+'\n')
            summary = dict(epoch=epoch, **aggregate(rows), true_groups_identity_sha256=identity,
                           prediction_binding_sha256=saved['binding_sha256'], provenance=dict(complete=cproof,metrics=mproof,predictions=pproof))
            report['epochs'].append(summary)
        except (ValueError, KeyError, TypeError, RuntimeError, EOFError) as error:
            report['issues'].append(dict(epoch=epoch,status='INVALID_OR_INCOMPLETE',error=str(error)))
    report['status'] = 'ALL_COMPLETED_EPOCHS_EVALUATED' if not report['issues'] else 'INCOMPLETE_OR_INVALID_EPOCHS_REPORTED'
    report['observed_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    fields = ['epoch'] + list(aggregate([dict(mixed_label=0,mixed_probability=.1)]).keys())
    with (out/'metrics.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(report['epochs'])
    print(json.dumps(dict(status=report['status'],output=str(out),evaluated_epochs=len(report['epochs']),issues=report['issues'],summary_sha256=digest((out/'summary.json').read_bytes())),allow_nan=False))
    return 0 if not report['issues'] else 2


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--through-epoch',type=int)
    args=parser.parse_args();return run(args.base,args.out,args.through_epoch)


if __name__ == '__main__':
    raise SystemExit(main())
