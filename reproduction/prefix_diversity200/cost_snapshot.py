"""Read only per-complete-outer native generation counts, never model tensors."""
import hashlib
import json
from pathlib import Path
import time


def collect(bases, run_id):
    rows = {}
    for name in bases:
        base = Path(name)
        if base.parent != Path('/mnt/data/shaoshitong') or not base.name.startswith('search-g1-prefix-diversity200-'):
            raise ValueError('Only this experiment remote namespaces are allowed')
        config = json.loads((base/'preparation/launch-config.json').read_text())
        if config['run_id'] != run_id:
            raise ValueError('Cost namespace has another logical run identity')
        for path in sorted((base/'outers').glob('outer-*/complete.json')):
            joint = json.loads(path.read_text()); outer = joint['outer']
            if joint['run_id'] != run_id or str(outer) in rows:
                raise ValueError('Repeated or foreign completed outer')
            prefix_path = path.parent/'prefix/complete.json'
            continuation_path = path.parent/'full/complete.json'
            prefix = json.loads(prefix_path.read_text()); continuation = json.loads(continuation_path.read_text())
            if prefix['trajectories'] != 25600 or continuation['trajectories'] != 5120:
                raise ValueError('Actual continuation budget differs')
            for value in (prefix, continuation):
                if value['run_id'] != run_id or value['outer'] != outer:
                    raise ValueError('Generation receipt identity differs')
                if value['actual_sampled_tokens'] != sum(x['actual_sampled_tokens'] for x in value['ranks']):
                    raise ValueError('Sampled token rank accounting differs')
                if value['native_generation_requests'] != sum(x['generation_requests'] for x in value['ranks']):
                    raise ValueError('Native request rank accounting differs')
            rows[str(outer)] = dict(outer=outer, base=str(base),
                prefix_sampled_tokens=prefix['actual_sampled_tokens'],
                continuation_new_sampled_tokens=continuation['actual_sampled_tokens'],
                combined_actual_sampled_tokens=prefix['actual_sampled_tokens']+continuation['actual_sampled_tokens'],
                prefix_native_generation_requests=prefix['native_generation_requests'],
                continuation_native_generation_requests=continuation['native_generation_requests'],
                actor_synchronized_adam_per_rank=20, probe_synchronized_adam_per_rank=20,
                rank_replicas_not_summed_as_global_steps=True, phase_timings=joint['phase_timings'],
                receipt_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (path,prefix_path,continuation_path)})
    return dict(run_id=run_id, observed_unix=time.time(), complete_outer_rows=rows,
        total_completed_outer_sampled_tokens=sum(x['combined_actual_sampled_tokens'] for x in rows.values()),
        cost_scope='successful committed outer generation only; failed attempts and native reservation wall are separate',
        retrieval_counts_pending_separate_remote_streaming_aggregation=True,
        model_tensors_read=False, new_model_forward=False, training_modified=False,
        source='existing native rank producer counts with independently read metadata SHA; raw tokens not re-read')
