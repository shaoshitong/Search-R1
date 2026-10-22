"""Remote CPU-only sampler experiment; audit RNG is not production E1 RNG."""
import hashlib
import importlib
import json
import random
import resource
import sys
import time
from pathlib import Path

BASE = Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/mixed-direct-e1-unique-g5-20261022-v1')
started = time.monotonic()
config = json.loads((BASE/'config.json').read_bytes())
publication = json.loads((BASE/'source-publication.json').read_bytes())
code = Path(publication['code_root'])
prefix = 'reproduction/operations/qwen7b_warmup/'
for name in ('unique_question_g5_sampler.py', 'balanced_hard_g10_sampler.py', 'g10_grouping.py'):
    relative = prefix+name
    data = (code/relative).read_bytes()
    assert len(data) == publication['files'][relative]['bytes']
    assert hashlib.sha256(data).hexdigest() == publication['files'][relative]['sha256']
sys.path.insert(0, str(code))
sampler_module = importlib.import_module('reproduction.operations.qwen7b_warmup.unique_question_g5_sampler')

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

reference = config['groups']
metadata = Path(reference['path']).read_bytes()
assert len(metadata) == reference['bytes'] and hashlib.sha256(metadata).hexdigest() == reference['sha256']
groups = json.loads(metadata)['train']
del metadata
audit_rng_seed = 9137
sampler = sampler_module.UniqueQuestionG5Sampler(groups,
    seed=config['unique_sampling_seed'], source_cursor=2577,
    epoch_end_cursors=tuple(config['epoch_end_cursors']),
    rng_state=random.Random(audit_rng_seed).getstate())
lookup = {group['qid']: group for group in groups}
epochs = []
orders = []
subsets = []
role_maps = []
for number, end_cursor in enumerate(config['epoch_end_cursors'], start=2):
    order = []
    by_qid = {}
    roles = {}
    updates = 0
    last_questions = None
    while sampler.accepted_updates < end_cursor:
        batch = sampler.next_batch()
        assert batch['grouping']['summary']['hard_pool_used'] is False
        for view, role in zip(batch['grouping']['groups'], batch['roles']):
            qid = view['qid']
            assert qid not in by_qid
            own = lookup[qid]
            indices = view['source_indices']
            assert len(indices) == len(set(indices)) == 5
            assert view['trajectory_ids'] == [own['trajectory_ids'][i] for i in indices]
            assert view['labels'] == [own['labels'][i] for i in indices]
            assert view['true_mixed'] == (0 < sum(view['labels']) < 5) == (role == 'mixed')
            order.append(qid)
            by_qid[qid] = tuple(indices)
            roles[qid] = role
        last_questions = len(batch['source_qids'])
        sampler.observe(batch, [0.] * last_questions)
        updates += 1
    assert len(order) == len(set(order)) == len(groups) == 169028
    counts = {role: sum(value == role for value in roles.values()) for role in ('mixed', 'nonmixed')}
    assert counts == {'mixed': 84514, 'nonmixed': 84514}
    assert updates == 1321 and last_questions == 68
    epochs.append(dict(epoch=number, questions=len(order), unique_questions=len(set(order)),
        class_counts=counts, updates=updates, tail_batch_questions=last_questions,
        ordered_question_ids_sha256=digest(order),
        own_G5_by_qid_sha256=digest(by_qid),
        role_by_qid_sha256=digest(roles),
        own_ids_and_EM_recomputed=True, hard_pool_used=False))
    orders.append(order)
    subsets.append(by_qid)
    role_maps.append(roles)
same_subsets = [q for q in subsets[0] if subsets[0][q] == subsets[1][q]]
same_order_position = sum(left == right for left, right in zip(*orders))
trajectory_overlap = sum(len(set(subsets[0][q]).intersection(subsets[1][q])) for q in subsets[0])
same_roles = sum(role_maps[0][q] == role_maps[1][q] for q in subsets[0])
assert epochs[0]['ordered_question_ids_sha256'] != epochs[1]['ordered_question_ids_sha256']
assert epochs[0]['own_G5_by_qid_sha256'] != epochs[1]['own_G5_by_qid_sha256']
result = dict(status='DEPLOYED_SAMPLER_ACTUAL169028_TWO_EPOCH_RANDOMIZATION_METADATA_PASSED',
    evidence_scope='Published deployed algorithm on actual accepted group metadata; audit-only deterministic RNG, not production E1 history',
    production_sampling_seed=config['unique_sampling_seed'], audit_inherited_rng_seed=audit_rng_seed,
    actual_production_E1_sampler_rng_read=False,
    sampler_publication_origin=publication['file_origins'][prefix+'unique_question_g5_sampler.py'],
    sampler_source=publication['files'][prefix+'unique_question_g5_sampler.py'],
    groups_reference=reference, epochs=epochs,
    comparison=dict(same_question_set_required_by_full_sweep=True,
        identical_question_order=False, same_question_order_positions=same_order_position,
        identical_G5_assignment=False, same_own_G5_questions=len(same_subsets),
        same_own_G5_fraction=len(same_subsets)/len(groups),
        different_own_G5_questions=len(groups)-len(same_subsets),
        same_role_questions=same_roles,
        mean_same_trajectory_ids_per_question=trajectory_overlap/len(groups),
        chance_repeated_G5_allowed=True, new_rollouts_generated=False),
    no_hard_replay=True, question_duplicates_within_each_epoch=0,
    all_169028_questions_in_each_epoch=True, GPU_used=False,
    model_checkpoint_feature_rollout_payload_opened=False,
    seconds=time.monotonic()-started, peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    epoch_time=time.time())
destination = BASE/'actual-metadata-two-epoch-randomization-audit.json'
raw = (json.dumps(result, sort_keys=True, separators=(',', ':'))+'\n').encode()
temporary = destination.with_suffix('.json.tmp')
temporary.write_bytes(raw)
temporary.replace(destination)
print(json.dumps(dict(result=result, receipt=dict(path=str(destination), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))))
