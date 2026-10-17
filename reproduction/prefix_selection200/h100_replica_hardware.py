"""Hardware-only placement for two independent four-rank reference replicas.

No training budget, objective, sampling parameter or data identity is set here.
Each experiment owns four cards; its exact FAISS index and reward services use
those same cards. Sharing with already authorized foreign jobs is recorded,
never treated as permission to stop them.
"""
LAYOUT = 'h100_four_rank_colocated_v1'
PREFIX = '/mnt/data/shaoshitong/search-g1-prefix-selection200-reference-replica-'


def enabled(config):
    return config.get('execution_hardware_layout') == LAYOUT


def validate(config):
    train = config['physical_gpu_indices']
    aux = config['auxiliary_gpu_indices']
    if (train not in ([0, 1, 2, 3], [4, 5, 6, 7]) or aux != train
            or config.get('retrieval_gpu_shards') != 4
            or config.get('grounding_physical_gpu') != train[-1]
            or config.get('grounding_physical_gpu_by_rank') != train
            or config.get('grounding_refit_visible_device') != train[-1]
            or config.get('world_size') != 4
            or len(config['physical_gpu_uuids']) != 4
            or len(set(config['physical_gpu_uuids'])) != 4
            or config['auxiliary_gpu_uuids'] != config['physical_gpu_uuids']
            or config.get('gpu_uuid_lock_root') != '/mnt/data/shaoshitong/.search-g1-gpu-uuid-locks'
            or config.get('shared_existing_foreign_jobs_authorized') is not True):
        raise ValueError('Bind one four-card slot with colocated exact-index retrieval and grounding')
    return config
