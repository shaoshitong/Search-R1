"""Read only per-complete-outer native generation counts, never model tensors."""
import hashlib
import json
import math
from pathlib import Path
import time


def grounding_binding(base, config, round_number, require_adoption=True):
    """Bind a reward generation to its committed manifest, without model loads."""
    base = Path(base)
    if round_number not in (0, 50, 100, 150):
        raise ValueError('Unknown reward-grounding round')
    if round_number == 0:
        path = Path(config['grounding_manifest'])
        version = config.get('grounding_initial_probe_version', config['grounding_probe_version'])
        expected_sha = config['grounding_manifest_sha256']
        completed = None
    else:
        completed = base/'grounding'/('round-%03d-complete.json' % round_number)
        receipt = json.loads(completed.read_text())
        if (receipt['run_id'] != config['run_id'] or receipt['round'] != round_number
                or not receipt['actual_both_heads_fitted']
                or receipt['actor_version'] != config['run_id']+'-actor%04d' % round_number):
            raise ValueError('Reward fit receipt belongs to another actual actor/round')
        path, version = Path(receipt['manifest']), receipt['version']
        expected_sha = receipt['manifest_sha256']
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if (hashlib.sha256(raw).hexdigest() != expected_sha
            or manifest['round'] != round_number or manifest['version'] != version
            or (round_number and (manifest['run_id'] != config['run_id']
                or manifest['actor_version'] != config['run_id']+'-actor%04d' % round_number))):
        raise ValueError('Committed reward manifest identity or SHA differs')
    adopted_path = base/'grounding'/('adopted-round-%03d.json' % round_number)
    if require_adoption and config.get('grounding_refit_rounds'):
        adopted = json.loads(adopted_path.read_text())
        if (adopted['round'] != round_number or adopted['version'] != version
                or adopted['manifest'] != str(path)):
            raise ValueError('Actually used reward version lacks matching adoption receipt')
    return dict(round=round_number, version=version, manifest=str(path),
        manifest_sha256=expected_sha, completed_receipt=str(completed) if completed else None,
        adoption_receipt=str(adopted_path) if adopted_path.exists() else None)


def collect(bases, run_id):
    rows = {}
    for name in bases:
        base = Path(name)
        if base.parent != Path('/mnt/data/shaoshitong') or not base.name.startswith('search-g1-prefix-diversity200-'):
            raise ValueError('Only this experiment remote namespaces are allowed')
        config = json.loads((base/'preparation/launch-config.json').read_text())
        if config['run_id'] != run_id:
            raise ValueError('Cost namespace has another logical run identity')
        reward_versions = {}
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
            reward = joint['actor']['reward']
            reward_round = min(150, ((outer-1)//50)*50) if config.get('grounding_refit_rounds') else 0
            if reward_round not in reward_versions:
                reward_versions[reward_round] = grounding_binding(base, config, reward_round)
            actual_binding = reward_versions[reward_round]
            if (reward['version'] != actual_binding['version']
                    or reward['trajectories'] != 2560 or reward['actual_reward_world_size'] != 8
                    or reward['prefix_correctness_head_used_for_reward']
                    or len(reward['ranks']) != 8
                    or {r['rank'] for r in reward['ranks']} != set(range(8))
                    or any(r['version'] != actual_binding['version'] for r in reward['ranks'])):
                raise ValueError('Actual grounding reward identity or scope differs')
            numbers = [reward['score_mean']] + [r[k] for r in reward['ranks']
                for k in ('score_mean', 'score_min', 'score_max')]
            if not all(math.isfinite(v) for v in numbers):
                raise ValueError('Nonfinite committed grounding reward statistics')
            rows[str(outer)] = dict(outer=outer, base=str(base),
                prefix_sampled_tokens=prefix['actual_sampled_tokens'],
                continuation_new_sampled_tokens=continuation['actual_sampled_tokens'],
                combined_actual_sampled_tokens=prefix['actual_sampled_tokens']+continuation['actual_sampled_tokens'],
                prefix_native_generation_requests=prefix['native_generation_requests'],
                continuation_native_generation_requests=continuation['native_generation_requests'],
                actor_synchronized_adam_per_rank=20, probe_synchronized_adam_per_rank=20,
                rank_replicas_not_summed_as_global_steps=True, phase_timings=joint['phase_timings'],
                actor_grounding_reward_summary=dict(version=reward['version'], trajectories=2560,
                    grounding_round=reward_round, grounding_binding=actual_binding,
                    score_mean=reward['score_mean'], score_min=min(r['score_min'] for r in reward['ranks']),
                    score_max=max(r['score_max'] for r in reward['ranks']),
                    rank_count=8, all_rank_statistics_finite=True,
                    prefix_correctness_head_used_for_reward=False, source='original saved actor reward receipt'),
                receipt_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (path,prefix_path,continuation_path)})
    return dict(run_id=run_id, observed_unix=time.time(), complete_outer_rows=rows,
        total_completed_outer_sampled_tokens=sum(x['combined_actual_sampled_tokens'] for x in rows.values()),
        cost_scope='successful committed outer generation only; failed attempts and native reservation wall are separate',
        retrieval_counts_pending_separate_remote_streaming_aggregation=True,
        model_tensors_read=False, new_model_forward=False, training_modified=False,
        source='existing native rank producer counts with independently read metadata SHA; raw tokens not re-read')
