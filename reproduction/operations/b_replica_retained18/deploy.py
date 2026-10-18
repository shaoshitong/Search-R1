"""Prepare independent B sources/config; execute only inside private namespace."""
import hashlib,json,pathlib,subprocess,time,uuid
from reproduction.prefix20_online.launch import once

def prepare(physical,canonical,source_config,commit):
    p=pathlib.Path(physical);c=json.loads(pathlib.Path(source_config).read_text());old=c['base']
    assert p.parent==pathlib.Path('/mnt/data/shaoshitong') and p.name.startswith('search-g1-prefix-selection200-reference-replica-b-own-')
    assert not (p/'preparation/launch-config.json').exists()
    def remap(v):
        if isinstance(v,str):return canonical+v[len(old):] if v==old or v.startswith(old+'/') else v
        if isinstance(v,dict):return {k:remap(x)for k,x in v.items()}
        if isinstance(v,list):return [remap(x)for x in v]
        return v
    c=remap(c)
    for key in ('startup_sampling_resume','node_migration','resume_import','resume_joint','zero_outer_cache_fault_source','zero_outer_cache_fault_proof_sha256','startup_fault_kind'):
        c.pop(key,None)
    run=uuid.uuid4().hex
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True).splitlines()
    mapping={int(x.split(',')[0]):x.split(',')[1].strip()for x in rows};cards=[4,5,6,7]
    c.update(base=canonical,run_id=run,execution_id=uuid.uuid4().hex,actor_version=run+'-official0',source_commit=commit,
        expected_execution_node='h100-10',physical_gpu_indices=cards,auxiliary_gpu_indices=cards,
        physical_gpu_uuids=[mapping[i]for i in cards],auxiliary_gpu_uuids=[mapping[i]for i in cards],
        retrieval_port=19510,search_url='http://127.0.0.1:19510/retrieve',grounding_probe_base_port=19520,
        actor_global_micro_batch_size=4,independent_B_replica=True,
        B_source_thread='01a0fcbb-863f-7450-926d-db537f648265',B_source_runtime='2e3976230ac9425d66d8553afa9962821e964932')
    for port in [19510,19520,19521,19522,19523]:
        import socket
        with socket.socket()as sock:sock.bind(('127.0.0.1',port))
    for name in ('preparation','logs'):(p/name).mkdir(exist_ok=True)
    once(p/'preparation/launch-config.json',c)
    once(p/'preparation/independent-B-identity.json',dict(run_id=run,physical=physical,canonical=canonical,source_config=source_config,source_commit=commit,prepared_unix=time.time(),fresh_actor=True,previous_B_state_imported=False,large_assets_via_Mac=False))
    return c
