"""Render small paired result metadata only; no model, rollout or final data."""
import json
from pathlib import Path

base = Path(__file__).resolve().parent
metrics = json.loads((base / 'actual-matched-metric-summary-complete.json').read_text())
assert metrics['status'] == 'NATURAL_VS_CONDITIONAL_G5_MATCHED_METRICS_COMPLETE'
assert metrics['missing'] == 0 and metrics['final4096_read'] is False
assert all(value == 0 for value in metrics['initial_logits'].values())
audits = json.loads((base / 'actual-CPU-checkpoint-audit-metadata-complete.json').read_text())
assert set(audits['audits']) == {'21', '27'}
lines = ['# Completed natural G5 paired control', '',
         'Original jobs 749459 (L21/P20) and 749460 (L27/P20) both completed with Slurm COMPLETED 0:0. '
         'Each trained three epochs, 192 real optimizer updates, global batch128 with two GPUs and eight accumulation microbatches. '
         'Actual total elapsed times were 10:20 and 12:40 respectively. No job was resubmitted or changed by this observer.', '',
         'The only intervention was label-blind uniform selection of five of each question’s own ten trajectories before deriving the real Mixed label. '
         'It restored 851/780/825 G10-Mixed→G5-homogeneous examples across the three epochs. '
         'All 8192 questions, epoch question order, frozen actor, native block/head initialization, online causal prefilling, '
         'LR5e-6, EMA0.99, optimizer budget, fixed training2048 and previously viewed independent validation2048 inputs match the conditional control. '
         'All eight initial raw/EMA layer/pool comparisons have exactly zero maximum logit difference.', '',
         'The tables compare saved predictions on identical question IDs, own G5 trajectories and labels. '
         'No final4096 labels, prediction scores or aggregates were read. These are previously viewed validation results, '
         'not fresh final-test results.', '']
for pool, title in [('fixed_validation2048', 'Previously viewed independent validation2048'),
                    ('fixed_train2048', 'Fixed training2048 diagnostic')]:
    lines += ['## ' + title, '',
              '| Layer/model | Epoch | Natural BCE | Natural Mixed BA | Natural true Mixed top512 | Conditional BCE | Conditional Mixed BA | Conditional true Mixed top512 |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in metrics['rows']:
        if row['pool'] != pool or row['epoch'] == 0:
            continue
        layer = row['layer'].replace('layer', '').replace('-prefix20', '')
        for model in ('raw', 'EMA'):
            natural = row['models'][model]['natural']
            conditional = row['models'][model]['conditional']
            ns, cs = natural['selected512'], conditional['selected512']
            lines.append('| L%s %s | %d | %.5f | %.2f%% | %d (%.2f%%) | %.5f | %.2f%% | %d (%.2f%%) |' %
                         (layer, model, row['epoch'], natural['binary_Mixed_BCE'], 100*natural['balanced_accuracy'],
                          ns['Mixed'], 100*ns['Mixed_fraction'], conditional['binary_Mixed_BCE'],
                          100*conditional['balanced_accuracy'], cs['Mixed'], 100*cs['Mixed_fraction']))
    lines += ['']
lines += ['## Interpretation', '',
          'The completed L21 natural run reaches its best top512 count of 204/512 (39.84% raw E3), below '
          'conditional L21’s 210/512 (41.02% raw E2). L27 natural raw E3 reaches 202/512 (39.45%), '
          'versus conditional E3’s 200/512 (39.06%); its EMA top512 result is lower (191 versus198). '
          'The natural sampling fix therefore does not show a consistent ranking gain in this pilot.', '',
          'At L27 raw E3, natural sampling reduces all-correct false alarms from 299/330 (90.61%) to253/330 (76.67%), '
          'and all-wrong false alarms from828/1055 (78.48%) to678/1055 (64.27%). '
          'It also increases real-Mixed misses from72/663 (10.86%) to150/663 (22.62%). '
          'This explains the modest balanced-accuracy improvement (53.88%→55.08%) while ranking barely changes. '
          'The full classwise counts and BCE, confusion matrices, recall, false-positive rates and paired deltas '
          'are preserved in actual-matched-metric-summary-complete.json.', '',
          '## Checkpoint acceptance', '']
for layer in ('21', '27'):
    report = audits['audits'][layer]
    assert report['status'] == 'NATURAL_G5_ONLINE_CPU_RAW_EMA_ADAM_RNG_CHECKPOINT_ACCEPTED'
    assert report['cursor'] == report['Adam']['all_actual_steps'] == 192
    assert report['raw_EMA_separate_and_different'] is True and len(report['RNG_by_rank']) == 2
    lines += ['- L%s: `%s`, %d bytes, SHA256 `%s`.' %
              (layer, report['checkpoint']['path'], report['checkpoint']['bytes'], report['checkpoint']['sha256'])]
lines += ['', 'Both actual final checkpoints fully deserialized on CPU. Raw and EMA are separate, finite and different '
          'in 46 native/head tensors; all Adam moments are finite and all clocks equal192; both rank Python/NumPy/CPU torch '
          'RNG states load correctly and CUDA RNG bytes are present. A GPU resume next-update replay was not performed or claimed. '
          'All checkpoint tensors, optimizer state, actor/model assets and input data remain remote.', '',
          'An HPC3 login-node restart briefly interrupted read access and shared-mount visibility. '
          'The jobs continued on their compute nodes; fresh accounting, final receipts and full checkpoint SHA acceptance '
          'after the incident establish completion. Missing remote paths were never recreated.', '',
          'The independent matched report has status NATURAL_VS_CONDITIONAL_G5_MATCHED_METRICS_COMPLETE, '
          'zero missing files and eight complete paired layer/epoch combinations. '
          'Its remote SHA and path are in actual-matched-comparison-complete.json.']
(base / 'RESULT-COMPLETE.md').write_text('\n'.join(lines) + '\n')
print(json.dumps({'status': 'SMALL_COMPLETED_CONTROL_REPORT_WRITTEN', 'path': str(base / 'RESULT-COMPLETE.md'),
                  'final4096_read': False}))
