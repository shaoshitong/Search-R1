"""Receiver-to-CPU acceptance only; GPU submission requires batched golden proof.

Reuses the published finite operator's guarded control functions.  No source
writer or running GPU task is altered by this separately identified operator.
"""
from advance_peer_newseed_feature_pipeline import call, event, wait_until

if __name__ == '__main__':
    wait_until(lambda value: value['receiver_complete'])
    call('ingest-prepare')
    call('ingest-start')
    state = wait_until(lambda value: bool(value['stages']['ingest'].get('complete')))
    event(dict(status='CPU_INGEST_ACCEPTED_WAITING_FOR_SEPARATE_BATCHED_GOLDEN_QUALIFICATION',
               state=state, gpu_jobs_submitted=False))
