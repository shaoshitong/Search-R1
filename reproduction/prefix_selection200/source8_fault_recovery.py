"""Same-state cross-node recovery after the source8 NFS service disappears.

A dead task in kernel exit may retain a CUDA context. It cannot run user code;
record that residue honestly and recover on a different physical node. Never
claim those contexts were released, or accept an ordinary sleeping writer.
"""
from pathlib import Path
from reproduction.prefix20_online.launch import read, sha

STATUS = 'SOURCE8_NFS_FAULT_NO_USERSPACE_WRITERS'


def exit_only(row):
    return row['state'] == 'Z' or (row['state'] == 'D' and bool(row['flags'] & 4))


def qualified(source):
    source = Path(source)
    path = source/'preparation/source8-fault-writer-barrier.json'
    proof = read(path)
    config = source/'preparation/launch-config.json'
    submit = read(source/'preparation/submit-receipt.json')
    if (proof['status'] != STATUS or proof['source_base'] != str(source)
            or proof['run_id'] != read(config)['run_id']
            or proof['config_sha256'] != sha(config)
            or proof['leader'] != submit['leader']
            or proof['source_node'] != 'h100-11'
            or proof['target_node'] != 'h100-5'
            or len(proof['checks']) != 2
            or any(c['leader_same_birth'] or
                   any(not exit_only(r) for r in c['remaining_owned_births'])
                   for c in proof['checks'])
            or not proof['authorization']['human_instruction']):
        raise ValueError('Cross-node recovery requires the actual no-writer fault barrier')
    return proof


def place_prepared(base, target_uuids):
    """Bind only placement after complete same-run import; no algorithm delta."""
    from reproduction.prefix20_online.launch import publish, once
    from .h100_replica_hardware import validate
    base = Path(base);path = base/'preparation/launch-config.json';c=read(path)
    source=Path(c['resume_joint']['source_base']);proof=qualified(source)
    if (base/'preparation/submit-intent.json').exists():
        raise ValueError('Never edit a submitted execution')
    if len(target_uuids)!=8 or len(set(target_uuids))!=8:
        raise ValueError('Need all actual target UUIDs')
    old=dict(expected_execution_node=c.get('expected_execution_node'),
             physical_gpu_uuids=c['physical_gpu_uuids'],auxiliary_gpu_uuids=c['auxiliary_gpu_uuids'])
    c.update(expected_execution_node=proof['target_node'],
             physical_gpu_uuids=[target_uuids[i] for i in c['physical_gpu_indices']],
             auxiliary_gpu_uuids=[target_uuids[i] for i in c['auxiliary_gpu_indices']])
    validate(c)
    record=dict(source_node=proof['source_node'],target_node=proof['target_node'],
                source_barrier=str(source/'preparation/source8-fault-writer-barrier.json'),
                source_barrier_sha256=sha(source/'preparation/source8-fault-writer-barrier.json'),
                previous=old,actual_target_uuids=target_uuids,science_changed=False)
    once(base/'preparation/source8-fault-node-placement.json',record)
    publish(path,c)
    return dict(base=str(base),config_sha256=sha(path),**record)
