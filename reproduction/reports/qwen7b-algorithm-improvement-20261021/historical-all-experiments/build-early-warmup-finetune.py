"""Compile existing small historical reports only; no evaluation or inference."""
import hashlib
import json
from pathlib import Path
import time

REPO = Path(__file__).resolve().parents[4]
OUTPUT = Path(__file__).resolve().parent
LOCAL = REPO.parent.parent / 'outputs/qwen7b-warmup-20261019'
REPORTS = REPO / 'reproduction/reports'
sources = {}

def read(path):
    path = path.resolve()
    raw = path.read_bytes()
    sources[str(path)] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    return json.loads(raw)

POOL = 'old unseen NQ-dev2048/G5, generated20261020; 641 Mixed; bankSHA5876bd9607b1e956d8752e13c6e54def5d77be683a940af5f03b4f0db176a64c; ordered qidsSHA02a3dad1b18eff01915dad654934510de148fa938658b40d32920104862baf0b'
NOTE = 'Independent Top512 pool is the old641-Mixed NQ-dev pool, NOT the later663-Mixed HF fixedG5 pool. fixed_diagnostic_BA is a separate512-question diagnostic. No cross-pool ranking is valid.'
rows = []

def append(experiment, checkpoint, epoch, metrics, source, top=None, independent_BA=None, train_pool=None, status='MEASURED_EXISTING_REPORT', extra_sources=None):
    train, diagnostic = metrics.get('train', {}), metrics.get('diagnostic', {})
    row = {'experiment': experiment, 'checkpoint': checkpoint, 'checkpoint_epoch': epoch, 'state': 'EMA0.99',
           'train_BA': train.get('mixed_balanced_accuracy'), 'test_BA': independent_BA,
           'top_mixed': top, 'top_k': 512 if top is not None else None,
           'pool_questions': 2048, 'pool_identity': POOL, 'train_pool': train_pool,
           'source': str(Path(source).resolve()), 'supporting_sources': extra_sources or [],
           'comparability_note': NOTE, 'status': status,
           'fixed_diagnostic_BA': diagnostic.get('mixed_balanced_accuracy'), 'fixed_diagnostic_questions': 512,
           'train_Mixed_BCE': train.get('group_bce'), 'train_trajectory_BCE': train.get('trajectory_bce'),
           'diagnostic_Mixed_BCE': diagnostic.get('group_bce'), 'diagnostic_trajectory_BCE': diagnostic.get('trajectory_bce')}
    rows.append(row)
    missing = dict(row)
    missing.update(state='raw', train_BA=None, test_BA=None, top_mixed=None, top_k=None,
                   fixed_diagnostic_BA=None, train_Mixed_BCE=None, train_trajectory_BCE=None,
                   diagnostic_Mixed_BCE=None, diagnostic_trajectory_BCE=None,
                   status='RAW_METRICS_MISSING_IN_EXISTING_LOCAL_REPORTS')
    rows.append(missing)

warmup_source = REPORTS / 'qwen7b-four-epoch-unseen2048-20261020/delivery/metrics-and-acceptance.json'
warmup = read(warmup_source)
selection = {r['epoch']: r for r in warmup['comparison']['results']}
derived_path = OUTPUT / 'old-unseen641-warmup-derived-BA.json'
derived = read(derived_path) if derived_path.exists() else None
derived_by_epoch = {r['epoch']: r for r in derived['results']} if derived else {}
for checkpoint in warmup['epochs']:
    epoch = checkpoint['epoch']
    assert checkpoint['completed_epoch'] and checkpoint['metrics']['train']['weight_kind'] == 'EMA0.99'
    append('7B native L26→block27 Prefix5 trajectoryBCE+MixedBCE warmup', checkpoint['path'], epoch,
           checkpoint['metrics'], warmup_source, selection[epoch]['selected512']['true_mixed'],
           independent_BA=derived_by_epoch.get(epoch, {}).get('mixed_balanced_accuracy'),
           train_pool='Fixed original169028 questions/G5 evaluation; fit uses233175 three-class-balanced groups with repeated whole G5s',
           extra_sources=[str(derived_path.resolve())] if derived else [])
    if derived:
        measured = rows[-2]
        measured['test_BA_origin'] = 'Authorized CPU-only confusion of original saved EMA q>=0.5 and own labels; no new forward'
        measured['test_confusion'] = derived_by_epoch[epoch]['confusion']
tail = warmup['preserved_tail_checkpoint_audit']['checkpoint']
append('7B native Prefix5 warmup E5 preserved tail ONLY', tail['path'], 5, {}, warmup_source,
       train_pool='Original169028/G5; only4 updates of E5 preserved after user reduced budget to4 full epochs',
       status='INCOMPLETE_E5_ONLY4_UPDATES_NO_COMPLETE_EPOCH_METRICS')

diagnosis_path = REPORTS / 'qwen7b-prefix20-balanced-rank1-20261020/mixed-error-diagnosis/mixed-ranking-diagnosis.json'
diagnosis = read(diagnosis_path)
old_independent = {r['epoch']: r['independent2048']['confusion_at_half']['balanced_accuracy'] for r in diagnosis['epochs']}
mixed_source = REPORTS / 'qwen7b-prefix20-mixed-20261020/epoch1-complete/prefix20-epoch1-complete-audit.json'
mixed = read(mixed_source)
mixed_config_path = REPORTS / 'qwen7b-prefix20-mixed-20261020/prefix20-current-source-config.json'
mixed_config = read(mixed_config_path)
assert mixed_config['config']['pair_weight'] == .1 and mixed_config['config']['prefix_tokens'] == 20
append('Mixed-only continuation from warmupE2, Prefix20, rank weight0.1', mixed['checkpoint']['path'], 1,
       mixed['metrics'], mixed_source, mixed['selected_mixed'], old_independent[1],
       train_pool='67199 genuineMixed G5 questions only; train BA is undefined because no negative class',
       extra_sources=[str(diagnosis_path.resolve()), str(mixed_config_path.resolve())])

sealed = {}
for path in sorted(LOCAL.glob('mixed-ft-status-*.json')):
    value = json.loads(path.read_text())
    checkpoint = value.get('latest.json', {})
    metric = value.get('metrics-latest.json', {})
    if '/balanced-rank1-from-epoch1-prefix20-20261020/probe/epoch-' not in checkpoint.get('path', ''):
        continue
    if not checkpoint.get('completed_epoch') or metric.get('epoch') != checkpoint.get('epoch'):
        continue
    epoch = checkpoint['epoch']
    if epoch not in sealed or value['time'] > sealed[epoch][0]['time']:
        sealed[epoch] = (value, path)
for epoch in sorted(sealed):
    value, path = sealed[epoch]
    value = read(path)
    metrics = value['metrics-latest.json']['metrics']
    independent_BA = old_independent.get(epoch)
    extra_sources = [str(diagnosis_path.resolve())] if independent_BA is not None else []
    audit_path = LOCAL / ('balanced-rank1-from-epoch1-prefix20-20261020-epoch%02d-prediction-audit.json' % epoch)
    if audit_path.exists():
        audit = read(audit_path)
        if 'selection_mixed_metrics' in audit:
            independent_BA = audit['selection_mixed_metrics']['mixed_balanced_accuracy']
            extra_sources.append(str(audit_path.resolve()))
    append('Prefix20 1:1 Mixed/nonMixed continuation, rank weight1.0', value['latest.json']['path'], epoch,
           metrics, path, metrics['selection2048']['selected512']['true_mixed'], independent_BA,
           train_pool='67198 unique G5 questions per epoch:33599Mixed+33599nonMixed; differs from allMixed E1 cohort',
           extra_sources=extra_sources)

prefix5_config_path = REPORTS / 'qwen7b-mixed-ft-cp2-20261020/five-epoch-alias-repair/mixed-current-source-config.json'
prefix5_failure_path = REPORTS / 'qwen7b-mixed-ft-cp2-20261020/five-epoch-alias-repair/mixed-ft-first-trial-failure.json'
prefix5_config = read(prefix5_config_path)
read(prefix5_failure_path)
assert prefix5_config['config']['prefix_tokens'] == 5
for label, epoch, cursor in [('E1', 1, None), ('E2', 2, None), ('CP256', None, 256)]:
    rows.append({'experiment': 'Original Prefix5 genuineMixed-only + auxiliary ranking from warmupE2',
                 'checkpoint': None, 'checkpoint_epoch': epoch, 'requested_checkpoint_tag': label,
                 'requested_cursor': cursor, 'state': 'unknown', 'train_BA': None, 'test_BA': None,
                 'top_mixed': None, 'top_k': None, 'pool_questions': 2048, 'pool_identity': POOL,
                 'train_pool': 'Planned67199 genuineMixed G5 questions; completed evaluated checkpoint not established locally',
                 'source': str(prefix5_config_path.resolve()), 'supporting_sources': [str(prefix5_failure_path.resolve())],
                 'comparability_note': 'Nested source_checkpoint.metrics belongs to warmupE2, NOT this fine-tune. '
                                       'No actual Prefix5 E1/E2/CP256 evaluation found in existing local records; do not substitute later Prefix20 metrics.',
                 'status': 'NO_LOCAL_ACCEPTED_PREFIX5_FINETUNE_CHECKPOINT_METRICS_FOUND'})

weighted = REPO / 'reproduction/reports/qwen7b-algorithm-improvement-20261021/mixed-direct-e1-uniform-g5-weighted/actual-status.json'
if weighted.exists():
    read(weighted)
    rows.append({'experiment': 'Uniform ownG5 class-weighted fork from direct E1 (later metadata only)',
                 'checkpoint': None, 'checkpoint_epoch': None, 'state': 'unknown', 'train_BA': None, 'test_BA': None,
                 'top_mixed': None, 'top_k': None, 'pool_questions': None, 'pool_identity': 'Not established by this small status file',
                 'train_pool': None, 'source': str(weighted.resolve()),
                 'comparability_note': 'This status reports no active same-name job, not completed evaluation metrics; do not infer success or raw/EMA state.',
                 'status': 'METRICS_MISSING_LOCAL_STATUS_ONLY'})

result = {'status': 'EXISTING_LOCAL_EARLY_7B_WARMUP_FINETUNE_HISTORY_COMPILED', 'time': time.time(),
          'rows': rows, 'sources': sources, 'new_evaluation_performed': False, 'final4096_read': False,
          'authorized_CPU_existing_prediction_summary_performed': derived is not None,
          'scope': '20261019/20 early warmup/Mixed-only/balanced rank1; later weighted fork missing-status inventory only',
          'limitations': ['Warmup completed4 full epochs; E5 is only a preserved4-update tail, not a fifth completed epoch.',
                          'Existing early reports evaluate EMA0.99; raw metrics are explicitly missing.',
                          'test_BA refers to independent2048 when actually reported; fixed512 diagnostic is a separate field.',
                          'WarmupE1..4 independent BA is supplied by authorized CPU confusion of original saved q/own labels; balancedE4/E5 independent BA remains missing locally.',
                          'Original Mixed-only Prefix5 launch attempts failed or were superseded before a completed reported evaluation.',
                          'This is history of evaluated probe Mixed discrimination, not downstream actor QA EM.']}
OUTPUT.mkdir(parents=True, exist_ok=True)
(OUTPUT / 'early-warmup-finetune.json').write_text(json.dumps(result, sort_keys=True) + '\n')
lines = ['# Early7B warmup and fine-tune history', '',
         'Only existing local small reports were read. All values below are actual recorded EMA0.99 results; raw metrics were not reported. '
         'The independent2048 pool here contains641 Mixed questions (31.2988%), distinct from the later663-Mixed HF fixedG5 pool. '
         'Do not rank results across those pools. No new forward or GPU evaluation was run. '
         'Warmup independent BA was added by authorized CPU-only counting of the original saved EMA q>=0.5 versus same-trajectory labels, '
         'with all original prediction SHA/bytes, bank/groups manifest, ordered qids and own G5 identity checked.', '',
         '| Experiment/checkpoint | Train Mixed BA | Fixed512 diagnostic BA | Independent2048 BA | Actual Mixed in top512 |',
         '| --- | ---: | ---: | ---: | ---: |']
def percent(value):return 'missing' if value is None else '%.4f%%' % (100*value)
for row in rows:
    if row['state'] == 'raw':continue
    top = 'missing' if row['top_mixed'] is None else '%d/%d' % (row['top_mixed'], row['top_k'])
    label = row['experiment'] + (' E%d' % row['checkpoint_epoch'] if row.get('checkpoint_epoch') is not None else '')
    lines.append('| %s (%s) | %s | %s | %s | %s |' %
                 (label, row['state'], percent(row['train_BA']),
                  percent(row.get('fixed_diagnostic_BA')), percent(row['test_BA']), top))
lines += ['', 'Mixed-only training has no negative class, so its train Mixed balanced accuracy is undefined. '
          'Its cohort differs from the later1:1 balanced67198-question training cohort. Warmup evaluated the original169028 questions, '
          'while fitting used a three-class-balanced233175-group plan with repeated wholeG5 groups.', '',
          'WarmupE5 was not completed: the user changed the target to4 full epochs and a4-update E5 tail was preserved. '
          'The balanced rank-weight1 continuation does have recorded completed E2/E3/E4/E5 metrics; these are continuation epoch numbers, '
          'not five additional balanced epochs.', '', '## Exact local sources', '']
for source, proof in sources.items():
    lines.append('- `%s` (%d bytes, SHA256 `%s`).' % (source, proof['bytes'], proof['sha256']))
lines += ['', 'Each JSON row also identifies the actual remote checkpoint path, pool identity, weight state, '
          'source and comparability limitations. Missing values remain null; no metric was borrowed from another weight state or later pool. '
          'The authorized warmup CPU derivation preserves the actual threshold0.5 confusion matrices in the JSON.']
(OUTPUT / 'early-warmup-finetune.md').write_text('\n'.join(lines) + '\n')
print(json.dumps({'rows': len(rows), 'actual_EMA_rows': sum(r['state']=='EMA0.99' and r['status']=='MEASURED_EXISTING_REPORT' for r in rows),
                  'output': str(OUTPUT), 'new_evaluation_performed': False}))
