"""Read existing refit/rollout JSON only; no inference or training side effects."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def counts(vector):
    require(isinstance(vector, list) and len(vector) == 5 and
            all(type(x) is int and x in (0, 1) for x in vector), 'Incomplete strict G5 labels')
    return sum(vector)


def summary(values):
    histogram = [values.count(k) for k in range(6)]
    size = len(values)
    return dict(questions=size, correct_count_histogram=histogram,
        mixed_questions=sum(histogram[1:5]),
        mixed_fraction=sum(histogram[1:5])/size if size else None,
        all_wrong_fraction=histogram[0]/size if size else None,
        all_correct_fraction=histogram[5]/size if size else None,
        rollout_em=sum(values)/(5*size) if size else None)


def compare(base, outer):
    require(outer in range(25, 200, 25), 'Only scheduled probe refits are comparable')
    phase = base/'artifacts/question-probe/periodic-refits'/str(outer)
    chunk_paths = [phase/('chunk-'+str(i))/'real-rollout-audit.json' for i in range(10)]
    output = dict(checked_unix=time.time(), outer=outer, selected_outer=outer+1,
                  status='WAITING_FOR_REFIT_RANDOM_LABELS', goal_complete=False)
    missing = [str(p) for p in chunk_paths if not p.is_file()]
    if missing:
        output['missing_chunks'] = len(missing)
        return output
    run = read(base/'logs/run-manifest.json')
    labels = []
    for path in chunk_paths:
        chunk = read(path)
        require(chunk['outer'] == outer and chunk['questions'] == 512 and
                chunk['trajectories'] == 2560 and len(chunk['labels']) == 512,
                'Wrong refit chunk boundary or budget')
        require(chunk['identity']['run_id'] == run['run_id'] and
                chunk['identity']['logical_update'] == outer and
                chunk['identity']['source_sha256'] == run['source_sha256'],
                'Refit chunk run/source binding differs')
        labels.extend(chunk['labels'])
    qids = {row['qid'] for row in labels}
    hashes = {row['question_hash'] for row in labels}
    require(len(qids) == len(hashes) == 5120, 'Refit labels are not distinct')
    values = [counts(row['strict_em']) for row in labels]
    require(all(k == row['correct_count'] for k, row in zip(values, labels)), 'Refit count differs')
    output.update(random_refit_sample=summary(values), source_chunk_sha256=[
        hashlib.sha256(p.read_bytes()).hexdigest() for p in chunk_paths])
    complete_path = phase/'complete.json'
    if not complete_path.is_file():
        output['status'] = 'WAITING_FOR_TWENTY_EPOCH_REFIT'
        return output
    complete = read(complete_path)
    require(complete['outer'] == outer and complete['fit_epochs'] == 20 and
            complete['groups'] == 5120 and complete['trajectories'] == 25600 and
            complete['head_optimizer_updates'] == 400 and complete['replay_groups'] == 0 and
            complete['head_adam_steps_before'] == 400*(outer//25) and
            complete['head_adam_steps_after'] == 400*(outer//25+1), 'Refit budget or Adam differs')
    require(all(complete[key] is False for key in ('student_updated', 'student_parameters_copied',
            'optimizer_reset', 'dynamic_feature_tensors_persisted')), 'Refit continuity differs')
    pre_path = Path(complete['pre_checkpoint_receipt'])
    require(hashlib.sha256(pre_path.read_bytes()).hexdigest() == complete['pre_checkpoint_sha256'],
            'Pre-refit checkpoint receipt SHA differs')
    pre = read(pre_path)
    require(pre['step'] == outer and pre['execution_phase'] == 'PROBE_PENDING' and
            len(pre['ranks']) == 4 and all(r['completed_outer'] == outer and
                r['actual_optimizer_writes'] == 20*outer for r in pre['ranks']),
            'Pre-refit checkpoint student boundary differs')
    found = []
    for path in (base/'artifacts/question-probe/attempts').glob('*/rollout-audit.json'):
        audit = read(path)
        if audit.get('step') == outer+1:
            found.append((path, audit))
    if not found:
        output['status'] = 'WAITING_FOR_POST_FIT_SELECTION_ROLLOUT'
        return output
    require(len(found) == 1, 'Multiple attempts need explicit selection; refusing to combine')
    path, audit = found[0]
    selection = read(path.with_name('selection.json'))
    require(selection['step'] == outer+1 and audit['source_commit'] == run['source_commit'] and
            audit['data_manifest_sha256'] == run['budget']['common_manifest_sha256'],
            'Selected source/data binding differs')
    online = selection['online_features']
    require(online['source'] == 'current_student_online' and
            online['student_completed_outer'] == outer and
            online['student_optimizer_writes'] == outer*20 and
            online['dynamic_tensors_persisted'] is False and
            online['student_gradient_enabled'] is False, 'Selected student boundary differs')
    require(audit['selected_groups'] == len(audit['groups']) == 512 and
            audit['raw_generated_trajectories'] == 2560 and audit['no_post_rollout_refill'] is True,
            'Selected rollout budget differs')
    require(selection['prediction_before_label'] is True and selection['post_rollout_em_filter'] is False and
            selection['policy_snapshot_id'].startswith(run['run_id']+':') and
            selection['head_version'] == audit['question_probe_version'], 'Selection provenance differs')
    selected = {row['qid']: row for row in selection['questions'] if row['selected']}
    require(len(selected) == 512 and set(selected) == {g['question_id'] for g in audit['groups']},
            'Selected question identities differ')
    actual = [counts(g['strict_correctness']) for g in audit['groups']]
    novel = [k for k, g in zip(actual, audit['groups']) if g['question_id'] not in qids and
             selected[g['question_id']]['question_hash'] not in hashes]
    predictions = [row['predicted_mixed_probability'] for row in selected.values()]
    require(all(isinstance(p, (float, int)) and math.isfinite(p) and 0 <= p <= 1
                for p in predictions), 'Invalid predicted probabilities')
    output.update(status='OBSERVED_RANDOM_AND_POST_FIT_SELECTED_G5_COMPARISON',
        selected=summary(actual), selected_outside_current_refit_pool_by_id_and_exact_text_sha=summary(novel),
        excluded_seen_questions=512-len(novel), mean_selected_predicted_mixed=sum(predictions)/512,
        difference_mixed_percentage_points=100*(summary(actual)['mixed_fraction']-
            output['random_refit_sample']['mixed_fraction']),
        refit_complete_sha256=hashlib.sha256(complete_path.read_bytes()).hexdigest(),
        selected_audit_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        scope='Same recorded student boundary; distinct question samples. Only the current refit pool is '
              'excluded from the outside-pool subset; earlier probe training pools are not excluded. '
              'No extra rollouts, held-out QA EM, '
              'all-candidate AUC, paired causal claim, or independently excluded canonical text variants.')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--outer', type=int, default=25)
    args = parser.parse_args()
    print(json.dumps(compare(args.base, args.outer), ensure_ascii=False, allow_nan=False))
