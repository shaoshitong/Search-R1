"""Post hoc influence and numerical acceptance of fixed V5 predictions.

This does not refit, select a different split, or replace the primary result.
Granular account identifiers remain remote; exported diagnostics are aggregates.
"""
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

import numpy as np

root=Path(sys.argv[1])
code=Path(sys.argv[2])
summary=json.loads((root/'results/summary.json').read_text())
cells=json.loads((root/'results/cells.json').read_text())
arrays=np.load(root/'results/granular_account_contributions.npz')
denominator=arrays['denominator']
base='source_fitted_quantity_demand_age'
transfer='A_account_transfer'
delta=arrays[transfer]-arrays[base]
order=np.argsort(denominator)[::-1]
active=denominator>0
loo=(delta.sum()-delta[active])/(denominator.sum()-denominator[active])
date_metrics=json.loads((root/'results/daily_metrics.json').read_text())
day_deltas={date:(date_metrics['losses'][transfer][date]-date_metrics['losses'][base][date])/total
            for date,total in date_metrics['totals'].items()}
influence={
    'post_hoc':True,'refitting_performed':False,
    'accounts_with_test_exposure':int(active.sum()),
    'accounts_with_negative_paired_loss_numerator':int(np.sum(delta[active]<0)),
    'accounts_with_positive_paired_loss_numerator':int(np.sum(delta[active]>0)),
    'top_10_account_exposure_share':float(denominator[order[:10]].sum()/denominator.sum()),
    'leave_one_account_out_delta_range':[float(loo.min()),float(loo.max())],
    'drop_highest_exposure_accounts':{str(k):float(delta[order[k:]].sum()/denominator[order[k:]].sum()) for k in (1,5,10)},
    'daily_paired_loss_differences':day_deltas,
    'interpretation':'Fixed predictions only. These diagnostics do not include model refitting or identify causal effects.'
}
fit_ranges={}
for index,name in enumerate(('alpha','beta','gamma')):
    values=np.array([cell['baseline_fit']['parameters'][index] for cell in cells])
    fit_ranges[name]={'min':float(values.min()),'median':float(np.median(values)),
                     'max':float(values.max()),'boundary_cells':sum(index in c['baseline_fit']['at_bound'] for c in cells)}
tuning={name:{str(power):sum(c['tuning'][name]['power']==power for c in cells) for power in (0.,.5,1.)}
        for name in cells[0]['tuning']}
observed=(root/'exit_code.txt').read_text().strip()
assert observed=='0' and len(cells)==60 and summary['sample']['split_seeds']==10
assert summary['optimization']['failed_baseline_fits']==summary['optimization']['failed_linear_fits']==0
assert summary['code_sha256']==hashlib.sha256(code.read_bytes()).hexdigest()
for model in summary['models']:
    assert np.isfinite(model['test_author_loss']) and np.isfinite(model['test_video_loss'])
    observed_loss=float(arrays[model['model']].sum()/denominator.sum())
    assert abs(observed_loss-model['test_author_loss'])<1e-10
report={'completed_utc':datetime.now(timezone.utc).isoformat(),
        'acceptance':{'exit_code':int(observed),'cells':len(cells),'source_code_sha_matches':True,
                      'all_reported_author_losses_reconstructed_from_saved_contributions':True},
        'fit_parameter_ranges':fit_ranges,'calibration_power_counts':tuning,'influence':influence,
        'source_files':{name:{'bytes':(root/'results'/name).stat().st_size,
              'sha256':hashlib.sha256((root/'results'/name).read_bytes()).hexdigest()}
              for name in ('summary.json','cells.json','granular_account_contributions.npz','daily_metrics.json')}}
(root/'results/diagnostics_and_acceptance.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
