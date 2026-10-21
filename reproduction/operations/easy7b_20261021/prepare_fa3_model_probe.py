"""Prepare a diagnostic-only native4 model comparison from a sealed source CP.

The existing production job is never stopped or edited by this entry. The
source checkpoint is selected explicitly once after this allocation starts;
the diagnostic does not dispatch training after its comparison.
"""
import argparse
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--source-base', required=True)
    args = parser.parse_args()
    from reproduction.easy2560_joint4.control import own_base, publish, restore_binding, sha
    from reproduction.easy2560_joint4.retrieval_proxy import external_binding
    from search_g1.easy2560_rolling_checkpoint import latest_checkpoint
    from search_g1.easy2560_restore import create_receipt
    base, source = own_base(args.base), own_base(args.source_base)
    if not os.environ.get('SLURM_JOB_ID', '').isdecimal() or base == source:
        raise ValueError('Require new private Slurm diagnostic base')
    submission = json.loads((base / 'preparation/model-probe-submit-claim.json').read_text())
    if submission['source_commit'] != args.commit or not submission['diagnostic_only']:
        raise ValueError('Diagnostic source submission differs')
    checkpoint = latest_checkpoint(source, verify=False)
    if checkpoint is None or not 9 <= checkpoint['step'] < 200:
        raise ValueError('Require actual sealed source checkpoint9 or later')
    event = dict(phase='READING_SEALED_SOURCE_FULL_STATE', source_checkpoint=checkpoint,
                 source_job=(source / 'job.id').read_text().strip(), diagnostic_only=True,
                 production_modified=False, checked_unix=time.time())
    publish(base / 'logs/model-probe-source-start.json', event)
    print(json.dumps(event), flush=True)
    receipt = base / 'preparation/model-probe-source-receipt.json'
    result = create_receipt(source, checkpoint['checkpoint'], receipt)
    print(json.dumps(dict(phase='SOURCE_RECEIPT_READY', receipt=result)), flush=True)
    external_receipt = base / 'preparation/external-retriever-receipt.json'
    external = external_binding(external_receipt)
    restoration = restore_binding(receipt)
    publish(base / 'preparation/training-submit-claim.json', dict(
        source_commit=args.commit, actor_micro_batch_per_rank=8,
        logical_micro_batch_per_rank=4, infra_profile='fa3-exact-cache-v1',
        external_retriever=external, **restoration, diagnostic_only=True,
        production_dispatch_allowed=False, created_unix=time.time()))
    from reproduction.easy2560_joint4.prepare_training import prepare
    prepared = prepare(base, args.commit, actor_micro=8, restore_receipt=str(receipt),
                       infra_profile='fa3-exact-cache-v1',
                       external_retriever_receipt=str(external_receipt))
    publish(base / 'logs/model-probe-prepared.json', dict(
        phase='NATIVE4_MODEL_COMPARISON_READY', prepared=prepared,
        source_receipt_sha256=sha(receipt), diagnostic_only=True,
        actual_full_state_cuda_restored=False, production_dispatch_allowed=False,
        checked_unix=time.time()))
    print('NATIVE4_MODEL_COMPARISON_READY', flush=True)


if __name__ == '__main__':
    main()
