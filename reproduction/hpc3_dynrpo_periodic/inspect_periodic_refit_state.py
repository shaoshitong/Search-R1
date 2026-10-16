"""Remote CPU-only, SHA-bound mmap inspection of periodic probe scalar metadata.

No model/Adam tensor bodies are accessed. This does not replace a complete
checkpoint restore test or prove GPU reload. It accepts only scheduled refits.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import time
import uuid


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def inspect(base, outer, include_minibatches=False):
    if outer not in range(25, 200, 25):
        raise ValueError('Only scheduled probe refits 25..175 are supported')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Explicitly set CUDA_VISIBLE_DEVICES empty for this CPU-only inspection')
    phase = base/'artifacts/question-probe/periodic-refits'/str(outer)
    pre_path = base/'artifacts/question-probe/pending-probe-pairs'/('global_step_'+str(outer))/'source-receipt.json'
    if not pre_path.is_file():
        return dict(checked_unix=time.time(), outer=outer, status='WAITING_FOR_PRE_REFIT_PAIR')
    pre_bytes = pre_path.read_bytes(); pre_sha = hashlib.sha256(pre_bytes).hexdigest()
    pre = json.loads(pre_bytes)
    assert pre['step'] == outer and pre['execution_phase'] == 'PROBE_PENDING'
    assert sorted(r['rank'] for r in pre['ranks']) == list(range(4))
    assert all(r['completed_outer'] == outer and r['actual_optimizer_writes'] == 20*outer for r in pre['ranks'])
    if not (phase/'latest-state.json').is_file() or not (phase/'latest-state.pt').is_file():
        return dict(checked_unix=time.time(), outer=outer, status='WAITING_FOR_EPOCH_RECOVERY_STATE', pre_receipt_sha256=pre_sha)
    import torch
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    assert not torch.cuda.is_initialized()
    holder = base/'diagnostics/phase-readback'; holder.mkdir(parents=True, exist_ok=True)
    start_steps = 400*(outer//25)
    for attempt in range(3):
        pointer = json.loads((phase/'latest-state.json').read_text())
        snapshot = holder/('reader-'+uuid.uuid4().hex+'.pt')
        try:
            os.link(phase/'latest-state.pt', snapshot)
            actual_sha = sha(snapshot)
            if actual_sha != pointer['sha256']:
                continue  # Atomic writer advanced between pointer and link; preserve the writer.
            saved = torch.load(snapshot, map_location='cpu', weights_only=False, mmap=True)
            p = saved['phase']
            assert saved['outer'] == pointer['outer'] == outer
            assert saved['pre_receipt_sha256'] == pre_sha
            assert p['epoch'] == pointer['epoch'] and p['batch'] == pointer['batch']
            assert 0 <= p['epoch'] <= 20 and 0 <= p['batch'] <= 20
            assert p['epoch'] < 20 or p['batch'] == 0
            assert p['start_steps'] == start_steps
            expected = start_steps+20*p['epoch']+p['batch']
            assert saved['head']['update_count'] == pointer['head_adam_steps'] == expected
            assert len(saved['indices']) == 5120
            label_count = len(saved['labels'])
            assert 0 <= label_count <= 5120 and label_count % 512 == 0
            if label_count < 5120:
                assert p['epoch'] == p['batch'] == 0 and saved['feature_hashes'] is None
            assert len(p['summaries']) == p['epoch']
            assert len(p['minibatches']) == 20*p['epoch']+p['batch']
            ranks = [{k:r[k] for k in ('rank','parameter_sha256','local_parameter_count',
                     'actual_optimizer_writes','completed_outer','scheduler_sha256','adam_steps')}
                     for r in saved['frozen_before']]
            assert sorted(r['rank'] for r in ranks) == list(range(4))
            assert all(r['actual_optimizer_writes'] == 20*outer and r['completed_outer'] == outer
                       and r['adam_steps'] == [20*outer] for r in ranks)
            assert not any(k in saved for k in ('features','feature_tensors','rollouts','raw_rollout'))
            out = dict(checked_unix=time.time(), status='SHA_BOUND_SCALAR_METADATA_VERIFIED', outer=outer,
                snapshot_sha256=actual_sha, pre_receipt_sha256=pre_sha, source_pointer=pointer,
                epoch=p['epoch'], batch=p['batch'], head_adam_steps=expected, start_adam_steps=start_steps,
                saved_questions=5120, saved_labels=label_count, labels_complete=label_count == 5120,
                completed_minibatches=len(p['minibatches']),
                summaries=p['summaries'], frozen_student_before=ranks,
                feature_hashes_present=saved['feature_hashes'] is not None,
                dynamic_tensor_fields_present=False, cuda_initialized=torch.cuda.is_initialized(),
                max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                scope='SHA-checked temporary hardlink and CPU mmap scalar metadata only; no model/Adam tensor-body read, inference or GPU reload; temporary link removed')
            if include_minibatches: out['minibatches'] = p['minibatches']
            del saved
            return out
        finally:
            snapshot.unlink(missing_ok=True)
    return dict(checked_unix=time.time(), outer=outer, status='RETRY_AFTER_ATOMIC_WRITER_PROGRESS', reader_retry_count=3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True, type=Path)
    parser.add_argument('--outer', required=True, type=int)
    parser.add_argument('--include-minibatches', action='store_true')
    args = parser.parse_args()
    print(json.dumps(inspect(args.base, args.outer, args.include_minibatches)))


if __name__ == '__main__':
    main()
