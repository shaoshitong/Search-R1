"""Read only the small actual driver and JSON recovery descriptors; no model weights."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    base = Path(args.base)
    assert base.parent == Path('/mnt/data/shaoshitong') and base.name.startswith('search-g1-prefix-diversity200-')
    import torch
    torch.set_num_threads(1)
    read = lambda p: json.loads(Path(p).read_text())
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    c = read(base/'preparation/launch-config.json')
    root = base/'outers/outer-000100'; joint = read(root/'complete.json')
    producer = joint['actor']['driver_state']; path = Path(producer['path'])
    assert path.is_relative_to(base) and path.stat().st_size == producer['bytes'] < 2*1024**2
    assert sha(path) == producer['sha256']
    before = path.stat(); saved = torch.load(path, map_location='cpu', weights_only=False)
    assert (before.st_ino, before.st_size, before.st_mtime_ns) == (path.stat().st_ino, path.stat().st_size, path.stat().st_mtime_ns)
    pool = read(joint['pool_snapshot']); reservation = read(root/'pool-reservation.json')
    def boundary(value):
        return {key: value.get(key) for key in ('next_outer','consumed_candidates','cycle','cursor','candidate_questions','seed')}
    actual_pool = saved['candidate_pool_state']
    previous_probe = saved['prefix_probe_receipt']
    expected_probe = read(base/'outers/outer-000099/complete.json')['probe']
    expected_manifest = read(root/'probe-score/manifest.json')
    result = dict(run_id=saved['run_id'], completed_outer=saved['completed_outer'],
        actor_version=saved['actor_version'], driver_state_keys=list(saved['driver_state']),
        driver_file=dict(path=str(path), bytes=before.st_size, sha256=producer['sha256']),
        config_sha256=sha(base/'preparation/launch-config.json'), config_source_commit=c['source_commit'],
        configured_pool_state_path=c.get('pool_state_path'), configured_prefix_probe_receipt=c.get('prefix_probe_receipt'),
        saved_driver_pool_path=actual_pool['path'], saved_driver_pool=boundary(actual_pool['value']),
        saved_driver_pool_sha256=actual_pool['sha256'],
        joint_pool_path=joint['pool_snapshot'], joint_pool=boundary(pool), joint_pool_sha256=sha(joint['pool_snapshot']),
        joint_pool_hash_matches_producer=sha(joint['pool_snapshot']) == joint['pool_sha256'],
        joint_pool_equals_reservation_after=pool == reservation['after'],
        driver_pool_equals_joint_pool=actual_pool['value'] == pool,
        saved_driver_previous_probe_path=previous_probe.get('path'),
        saved_driver_previous_probe={k: previous_probe['value'].get(k) for k in ('completed_outer','actual_adam_steps_per_rank','sha256','run_id')},
        expected_previous_probe={k: expected_probe.get(k) for k in ('completed_outer','actual_adam_steps_per_rank','sha256','run_id')},
        score_manifest_previous_probe_sha256=expected_manifest['probe_before_checkpoint_sha256'],
        driver_previous_probe_matches_score=previous_probe['value']['sha256'] == expected_manifest['probe_before_checkpoint_sha256'],
        expected_previous_probe_matches_score=expected_probe['sha256'] == expected_manifest['probe_before_checkpoint_sha256'],
        driver_python_numpy_cpu_rng_present=all(key in saved['rng'] for key in ('python','numpy','torch_cpu','torch_cuda')),
        training_modified=False, large_weight_files_read=False)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__': main()
