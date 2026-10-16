"""Plot only the SHA-bound actual200 scalar and dev512 summaries."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parents[2] / 'outputs/tree-nested-branch-handoff-20261013'
INPUTS = {
    'LEARNING_CURVE_COMPLETE_THROUGH200_ORIGINAL_RECORDS.json':
        '16732748cf740c264ad707e26070bd5386ab562b551354ce1a23e9ca63226df5',
    'METRIC_RECORD_COVERAGE_THROUGH200_ORIGINAL_RECORDS.json':
        '37b80e80f7318faa66d81ac316d4078405fc895c1ea3038ad51c9a9b808944e3',
}
loaded = {}
for name, expected in INPUTS.items():
    raw = (BASE / name).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == expected
    loaded[name] = json.loads(raw)
curve = loaded[next(iter(INPUTS))]
coverage = loaded['METRIC_RECORD_COVERAGE_THROUGH200_ORIGINAL_RECORDS.json']['coverage']
series = curve['complete_original_training_series200']
assert len(curve['training_points']) == 200
dev = curve['completed_dev512_points']
assert [p['outer'] for p in dev] == list(range(5, 201, 5))
assert all(p['rows'] == 512 for p in dev)
x = np.arange(1, 201)
for values in series.values():
    assert len(values) == 200 and np.isfinite(values).all()

plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.dpi': 150})
fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), layout='constrained')
a = axes[0, 0]
a.plot([p['outer'] for p in dev], [100*p['f1'] for p in dev], label='dev512 F1')
a.plot([p['outer'] for p in dev], [100*p['em'] for p in dev], label='dev512 EM')
a.set(title='Fixed dev512: 40 completed evaluations', ylabel='Score (%)')
a.legend()
a = axes[0, 1]
a.plot(x, series['raw_within_tree_reward_population_variance'], label='Within-root reward variance')
a.plot(x, series['pure_intra_adv_population_variance'], label='Pure intra advantage variance')
a.plot(x, series['total_actual_adv_population_variance'], label='Total advantage variance')
a.set(title='Recorded population variances', ylabel='Variance')
a.legend(fontsize=8)
a = axes[1, 0]
for key, label in [('all_reward_one_question_count', 'Five leaves all correct'),
                   ('all_reward_zero_question_count', 'Five leaves all wrong')]:
    values = coverage[key]['values_in_outer1_to200_order']
    assert len(values) == 200 and not coverage[key]['unknown_outers']
    a.plot(x, np.array(values)/512*100, label=label)
a.plot(x, np.array(series['mixed_reward_question_fraction'])*100, label='Mixed rewards')
a.set(title='Five-leaf question outcomes (512 questions/outer)', ylabel='Questions (%)')
a.legend(fontsize=8)
a = axes[1, 1]
a.plot(x, np.array(series['training_reward_mean'])*100, label='Training reward mean')
a.plot(x, np.array(series['mixed_reward_tree_fraction'])*100, label='Mixed-reward roots')
a.set(title='Training signal and remaining reward uniformity', ylabel='Percent')
a.legend(fontsize=8)
for a in axes.flat:
    a.set(xlabel='Completed outer update', xlim=(1, 200))
    a.grid(alpha=.18)
fig.suptitle('Reward-selected nested Tree-GRPO: actual 200 updates', fontsize=14)
fig.supxlabel('A: 2 leaves; B: 3 leaves. Selected-data surrogate; no matched GRPO comparison. Final 51,713 scores pending.', fontsize=9)
outputs = []
for suffix in ('png', 'pdf'):
    path = BASE / ('ACTUAL200_TRAINING_AND_VARIANCE.' + suffix)
    assert not path.exists()
    fig.savefig(path)
    raw = path.read_bytes()
    outputs.append(dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
receipt = dict(schema='actual200_scientific_plot_v1', source_inputs=INPUTS,
               training_points=200, dev_points=40, final51713_complete=False,
               smoothing=False, interpolation=False, matched_control_claimed=False,
               outputs=outputs)
path = BASE / 'ACTUAL200_TRAINING_AND_VARIANCE_PLOT_RECEIPT.json'
assert not path.exists()
path.write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt))
