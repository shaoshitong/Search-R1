"""CPU-only native FSDP conversion while the GPU evaluation is explicitly held."""
import fcntl
import json
import os
import subprocess
import time
from common import OUT, TRAIN, atomic, sha
from run import merge_role


def main():
    assert os.environ.get('SLURM_JOB_ID')
    gpu_job=(OUT/'job.id').read_text().strip()
    state=subprocess.check_output(['scontrol','show','job',gpu_job],text=True)
    assert 'JobState=PENDING' in state and 'Priority=0' in state
    lock=(OUT/'eval.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=json.loads((OUT/'source-manifest.json').read_text())
    for rel,digest in manifest['files'].items():
        assert sha(OUT/rel)==digest,rel
    saved=json.loads((TRAIN/'checkpoint-receipt-200.json').read_text())
    assert saved['step']==200 and saved['status']=='BOTH_ROLE_FULL_CHECKPOINT_FILES_SHA_PASS'
    for role,name in [('actor','main'),('reranker_actor_rollout','reranker')]:
        receipt=OUT/('export-%s-verified.json'%name)
        if receipt.exists():
            proof=json.loads(receipt.read_text())
            for item in proof['input_files']:
                assert sha(item['path'])==item['sha256']
            for filename,digest in proof['files'].items():
                assert sha(OUT/'models'/name/filename)==digest
        else:
            merge_role(role,name,saved)
    atomic(OUT/'cpu-model-preparation-complete.json',dict(
        status='BOTH_NATIVE_FSDP200_MODELS_EXPORTED_READBACK_SHA_PASS',
        epoch=time.time(),job=int(os.environ['SLURM_JOB_ID']),gpus_requested=0,
        source_manifest_sha256=sha(OUT/'source-manifest.json')))
    print('BOTH_ROLES_CPU_PREPARATION_COMPLETE',flush=True)


if __name__=='__main__':
    main()
