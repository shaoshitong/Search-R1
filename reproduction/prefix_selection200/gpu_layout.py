"""Explicit physical placement; sharing never changes the four training ranks."""
SHARED = 'four_shared_train_retrieval_v1'


def placement(c):
    train = c.get('physical_gpu_indices', [])
    aux = c.get('auxiliary_gpu_indices', [])
    training_uuids = c.get('physical_gpu_uuids', [])
    aux_uuids = c.get('auxiliary_gpu_uuids', [])
    if any(len(v) != 4 or len(set(v)) != 4 for v in (train, aux, training_uuids, aux_uuids)):
        raise ValueError('Exactly four distinct training ranks and four retrieval shards required')
    if c.get('gpu_resource_layout') == SHARED:
        if (aux != train or aux_uuids != training_uuids
                or c.get('grounding_physical_gpus') != train
                or c.get('grounding_physical_gpu') != train[-1]
                or c.get('grounding_refit_visible_device') != train[-1]):
            raise ValueError('Shared layout must bind retrieval and one grounding replica to each training GPU')
    elif c.get('gpu_resource_layout') is not None:
        raise ValueError('Unknown physical GPU layout')
    elif set(train) & set(aux) or len(set(training_uuids + aux_uuids)) != 8:
        raise ValueError('Legacy separate auxiliary layout requires eight distinct GPUs')
    return list(dict.fromkeys(train + aux)), list(dict.fromkeys(training_uuids + aux_uuids))


def grounding_device(c, rank):
    return c['grounding_physical_gpus'][rank] if c.get('gpu_resource_layout') == SHARED else c['grounding_physical_gpu']
