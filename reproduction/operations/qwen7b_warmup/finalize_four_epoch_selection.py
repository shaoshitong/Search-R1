"""Final acceptance for the newer human FOUR-epoch + unseen2048 scope.

The immutable original config still says20 because every recovery checkpoint
binds its SHA. The newer direct human request changes the effective budget.
Never fabricate FULL_DATA_7B_PROBE_WARMUP20_COMPLETE or run the old20 gate.
"""
import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import unicodedata


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def normalized(value):
    return ' '.join(unicodedata.normalize('NFKC', value).casefold().strip().rstrip('?').split())


def terminal(job):
    live = subprocess.check_output(['squeue', '-h', '-j', str(job), '-o', '%i|%T'], text=True)
    assert not live.strip(), live
    account = subprocess.check_output(['sacct', '-X', '-n', '-P', '-j', str(job), '--format=JobID,State,ExitCode'], text=True)
    rows = [line.strip().split('|') for line in account.splitlines() if line.strip()]
    assert rows and all(row[1:3] == ['COMPLETED', '0:0'] for row in rows), account
    return account


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--check-ready', action='store_true')
    args = parser.parse_args()
    base, evaluation = args.base, args.base / 'selection-2048-20261020'
    required = [base / 'control/epoch4-stop-complete.json', evaluation / 'control/pipeline-produced.json',
                evaluation / 'results/comparison.json', evaluation / 'results/recommended-ema.json']
    required += [base / 'probe' / ('epoch-%04d.json' % epoch) for epoch in range(1, 5)]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        print(json.dumps(dict(status='NOT_READY', effective_fit_epochs=4, missing=missing, goal_complete=False)))
        return
    if args.check_ready:
        print(json.dumps(dict(status='ARTIFACTS_PRESENT_FINAL_ACCEPTANCE_STILL_REQUIRED', goal_complete=False)))
        return
    request_path = base / 'control/human-epoch4-selection-request.json'
    request, config = read(request_path), read(evaluation / 'config.json')
    assert request['source'] == 'latest_direct_human_message' and request['enabled'] and request['stop_after_epoch'] == 4
    assert sha(request_path) == config['human_request_sha256']
    stop = read(base / 'control/epoch4-stop-complete.json')
    assert stop['request_sha256'] == config['human_request_sha256'] and stop['epoch4']['steps'] == 18292
    sampling = read(evaluation / 'control/sampling-submission.json')['job_id']
    scoring = read(evaluation / 'control/score-submission.json')['job_id']
    accounting = dict(fit=terminal(740358), sampling=terminal(sampling), scoring=terminal(scoring))
    auditors = base / 'auditors/fe0e65058ff3dba9deec831f9fc8dec18153caea'
    sys.path.insert(0, str(auditors))
    import finalize as original
    assert Path(original.__file__).resolve().parent == auditors.resolve()
    training_config, bank, groups, source, recovery = original.verify_inputs(base)
    from reproduction.operations.qwen7b_warmup.train import balanced_plan
    plans = [balanced_plan(groups['train'], training_config['seed'], epoch)[0] for epoch in range(1, 5)]
    assert all(len(plan) == 4573 and len(set(i for batch in plan for i in batch)) == 169028 for plan in plans)
    count, tail = 0, []
    with (base / 'probe/curve.jsonl').open() as handle:
        for line in handle:
            row = json.loads(line)
            count += 1
            assert row['steps'] == row['ema_steps'] == count and row['actor_updates'] == 0
            assert all(math.isfinite(row[key]) and row[key] >= 0 for key in ('trajectory_bce', 'group_bce', 'total_loss'))
            if count <= 18292:
                epoch = (count - 1) // 4573 + 1
                assert row['epoch'] == epoch and row['batch'] == (count - 1) % 4573 + 1 and row['batches'] == 4573
            else:
                assert row['epoch'] == 5 and row['batch'] == count - 18292
                tail.append(row)
    assert count >= 18292
    assert read(base / 'probe/latest.json')['steps'] == count
    tail_proof = None
    if tail:
        subprocess.check_call([sys.executable, str(auditors / 'audit_checkpoint.py'), '--base', str(base),
                               '--receipt', str(base / 'probe/latest.json')])
        tail_proof = read(base / 'checkpoint-audit' / ('step-%08d.json' % count))
        assert tail_proof['steps'] == count and tail_proof['checked_ranks'] == 4
    epoch_receipts, prediction_proofs = [], []
    for epoch in range(5):
        proof_path = base / 'prediction-audit' / ('epoch-%04d.json' % epoch)
        if not proof_path.exists():
            subprocess.check_call([sys.executable, str(auditors / 'audit_predictions.py'), '--base', str(base), '--epoch', str(epoch)])
        proof = read(proof_path)
        assert proof['status'] == 'FULL_NATURAL_G5_EPOCH_PREDICTIONS_AND_METRICS_VERIFIED'
        assert sha(proof['metric_file']) == proof['metric_file_sha256']
        for split in ('train', 'diagnostic'):
            assert proof['splits'][split]['groups'] == (169028 if split == 'train' else 512)
            for item in proof['splits'][split]['files']:
                assert Path(item['path']).stat().st_size == item['bytes'] and sha(item['path']) == item['sha256']
        prediction_proofs.append(proof)
        if not epoch:
            continue
        receipt = read(base / 'probe' / ('epoch-%04d.json' % epoch))
        assert receipt['completed_epoch'] and receipt['epoch'] == epoch and receipt['steps'] == epoch * 4573
        assert Path(receipt['path']).stat().st_size == receipt['bytes'] and sha(receipt['path']) == receipt['sha256']
        cpu = read(base / 'checkpoint-audit' / ('step-%08d.json' % receipt['steps']))
        assert cpu['status'] == 'INDEPENDENT_CPU_FULL_CHECKPOINT_SHA_MODELS_ADAM_FOUR_RANK_RNG_VERIFIED'
        assert cpu['checkpoint']['sha256'] == receipt['sha256'] and cpu['checked_ranks'] == 4 and cpu['optimizer_parameter_tensors'] == 14
        epoch_receipts.append(receipt)
    # Verify the fixed unseen questions against the untouched exclusion sets.
    common = Path(training_config['canonical_train']).parent
    acceptance = read(evaluation / 'preparation/question-plan-acceptance.json')
    excluded_ids, excluded_questions = set(), set()
    for split in ('train', 'validation', 'stage', 'final'):
        path = common / (split + '.jsonl')
        assert sha(path) == acceptance['excluded_jsonl_sha256'][split]
        for line in path.open():
            row = json.loads(line)
            excluded_ids.add(str(row['question_id']))
            excluded_questions.add(normalized(row['question']))
    raw_path = evaluation / 'source/nq-dev-bcafb8dd07d4.jsonl'
    assert sha(raw_path) == config['heldout_source_sha256'] == '10d5ee25e813f3e2092247a5d2797793dd47ef5853959989cb34bf80a110d607'
    raw = [json.loads(line) for line in raw_path.open()]
    frozen = read(evaluation / 'inputs/questions.json')['rows']
    assert len(raw) == 8757 and len(frozen) == 2048
    selected_text = set()
    for row in frozen:
        own = raw[row['source_ordinal']]
        assert row['source_question_id'] == 'nq:dev:' + str(own['id'])
        assert row['source_question_id'] not in excluded_ids and normalized(own['question']) not in excluded_questions
        assert row['reward_model']['ground_truth']['target'] == own['golden_answers']
        selected_text.add(normalized(own['question']))
    assert len(selected_text) == 2048
    unseen = read(evaluation / 'bank/complete.json')
    assert unseen['status'] == 'UNSEEN_2048_G5_NATIVE_FEATURES_OWN_EM_VERIFIED'
    assert unseen['questions'] == 2048 and unseen['trajectories'] == 10240 and unseen['actor_updates'] == 0
    for item in unseen['batches']:
        assert sha(item['path']) == item['sha256']
        batch = read(item['path'])
        for stage, keys in ((batch['prefix'], ('states', 'prefix', 'actor_hidden')), (batch['full'], ('full', 'own_em'))):
            for key in keys:
                assert sha(stage[key]['path']) == stage[key]['sha256']
        for item in read(batch['prefix']['actor_hidden']['path'])['files'].values():
            assert Path(item['path']).stat().st_size == item['bytes'] and sha(item['path']) == item['sha256']
    # Recompute predicted-only rankings and real Mixed from saved probabilities;
    # never rerun write-mode merge, whose bank SHA is already checkpoint-bound.
    publication = read(evaluation / 'control/source-publication.json')
    score_script = Path(publication['files']['heldout_selection.py']['path'])
    assert sha(score_script) == publication['files']['heldout_selection.py']['sha256']
    subprocess.check_call([sys.executable, str(score_script), '--mode', 'report', '--base', str(evaluation)])
    comparison = read(evaluation / 'results/comparison.json')
    assert comparison['status'] == 'FOUR_EMA_UNSEEN2048_TOP512_TRUE_MIXED_COMPARISON_VERIFIED'
    for controller in (base / 'control/epoch4-stop-controller.json', evaluation / 'control/coordinator-controller.json'):
        item = read(controller)
        proc = Path('/proc') / str(item['pid'])
        matches = proc.exists() and proc.stat().st_uid == os.getuid() and proc.joinpath('stat').read_text().rsplit(')', 1)[1].split()[19] == item['start_ticks']
        assert not matches, 'Owned controller has not exited yet: ' + str(controller)
    for path in (base / 'probe/fit.lock', base / 'control/epoch4-stop.lock',
                 evaluation / 'control/coordinator.lock', evaluation / 'control/scoring.lock'):
        assert path.exists()
        with path.open('a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(handle, fcntl.LOCK_UN)
    output = base / 'delivery-four-epoch-selection'
    output.mkdir(exist_ok=True)
    baseline = read(base / 'probe/epoch-0000-metrics.json')
    rows = [dict(epoch=0, **baseline)] + [dict(epoch=item['epoch'], **item['metrics']) for item in epoch_receipts]
    original.plot_curves(rows, output / 'epoch-curves.png')
    fields = ['epoch', 'split', 'trajectory_bce', 'group_bce', 'total_loss', 'trajectory_accuracy', 'mixed_balanced_accuracy']
    with (output / 'epoch-curves.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            for split in ('train', 'diagnostic'):
                writer.writerow(dict(epoch=row['epoch'], split=split, **{key: row[split][key] for key in fields[2:]}))
    report = ['# Four-epoch native7B probe warmup and unseen selection', '',
              'Newer direct human instruction changed the old20-epoch budget to FOUR complete epochs, then independent2048-question validation.',
              'Actor remained frozen; source, all train/dev G5, native features, GPU restore proof and four complete Adam/EMA/RNG states accepted.',
              'Original immutable config still states20 for checkpoint SHA compatibility; no20-epoch completion is claimed.', '',
              'Actual completed requested updates:18292. Preserved graceful-stop epoch5 tail updates:%d.' % len(tail),
              'All artifacts and source data stay remote. The reserved51713 final questions were not used for fitting or selection.', '',
              (evaluation / 'results/REPORT.md').read_text(), '',
              'Recommended full checkpoint: `%s`, weight key `ema`.' % comparison['recommended_checkpoint'], '',
              '![Warmup curves](epoch-curves.png)']
    (output / 'REPORT.md').write_text('\n'.join(report))
    result = dict(status='HUMAN_FOUR_EPOCH_WARMUP_AND_UNSEEN2048_SELECTION_ACCEPTED',
                  effective_fit_epochs=4, original_twenty_epoch_budget_superseded=True,
                  human_request_sha256=sha(request_path), completed_requested_updates=18292,
                  preserved_tail_updates=len(tail), accounting=accounting, epochs=epoch_receipts,
                  preserved_tail_checkpoint_audit=tail_proof,
                  prediction_audits=prediction_proofs, recovery_proof=recovery,
                  selected_question_plan=acceptance, comparison=comparison,
                  recommended_epoch=comparison['recommended_epoch'], actor_updates=0,
                  finalizer_source_sha256=sha(__file__), finalizer_path=str(Path(__file__).resolve()), time=time.time())
    (output / 'metrics-and-acceptance.json').write_text(json.dumps(result, indent=2))
    files = {name: dict(bytes=(output / name).stat().st_size, sha256=sha(output / name))
             for name in ('REPORT.md', 'epoch-curves.png', 'epoch-curves.csv', 'metrics-and-acceptance.json')}
    (output / 'complete.json').write_text(json.dumps(dict(status=result['status'], files=files,
         effective_fit_epochs=4, recommended_epoch=result['recommended_epoch'],
         original_twenty_epoch_goal_not_claimed_complete=True, time=time.time()), indent=2))
    print('FINAL_RESULT_JSON=' + json.dumps(dict(status=result['status'], output=str(output),
          recommended_epoch=result['recommended_epoch'], preserved_tail_updates=len(tail), files=files)))


if __name__ == '__main__':
    main()
