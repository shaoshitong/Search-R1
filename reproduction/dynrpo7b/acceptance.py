"""Read actual iteration and checkpoint results; no synthetic GPU workload."""
import json
import math
from pathlib import Path
import time

from .mixed_backstop import _atomic_json


def accept_first_five(base,checkpoint,records):
    base=Path(base)
    run=json.loads((base/'logs/run-manifest.json').read_text())
    complete=json.loads((Path(checkpoint)/'complete.json').read_text())
    if complete['run_id']!=run['run_id'] or complete['step']!=5:
        raise ValueError('First-five acceptance requires its actual fifth complete checkpoint')
    if [r['step'] for r in records]!=[1,2,3,4,5]:
        raise ValueError('Five complete consecutive outers have not been observed')
    for row in records:
        metrics=row['metrics']
        for key in ('critic/rewards/mean','actor/pg_loss','actor/grad_norm'):
            if not math.isfinite(metrics[key]):
                raise ValueError('Nonfinite actual training metric: '+key)
        writes=metrics['actor/optimizer_steps_this_update']
        opportunities=metrics['actor/update_opportunities_this_attempt']
        if not 1<=writes<=20 or opportunities!=20:
            raise ValueError('A complete outer must process all20 opportunities with actual Adam writes')
    value=dict(accepted=True,run_id=run['run_id'],actual_updates=records,
        checkpoint=complete,checked_unix=time.time(),
        scope='five actual actor outers, finite training metrics and full four-rank checkpoint',
        separate_synthetic_GPU_tests=False,continuous_GPU_sampler=False)
    _atomic_json(base/'logs/five-updates-acceptance.json',value)
    print('DYNRPO_FIRST_FIVE_COMPLETE='+run['run_id'],flush=True)
    return value
