"""Sole-root local state ingestion from an existing read-only native observer."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path


def ingest(state_path, directory):
    state_path, directory = Path(state_path), Path(directory)
    summary_path = directory / 'actorhidden-period20-latest-compact.json'
    state = json.loads(state_path.read_text())
    summary = json.loads(summary_path.read_text())
    raw_path = Path(summary['last_original'])
    raw = json.loads(raw_path.read_text())
    if (str(summary['job']) != str(state['current_job_id'])
            or raw['base'] != state['base']
            or (raw.get('stage') or {}).get('run_id') not in (None, state['run_id'])):
        raise ValueError('A stale observer cannot overwrite another actual run')
    complete = raw['complete_outer']
    if type(complete) is not int or complete < state['complete_outer']:
        raise ValueError('Completed native evidence must not move backwards')
    if complete:
        if (raw['actor_adam_per_rank'] != [20 * complete] * 8
                or raw['probe_adam_per_rank'] != [100 * (complete // 20)] * 8):
            raise ValueError('Actual completed actor/probe clocks differ')
        state['actor_adam_per_rank'] = raw['actor_adam_per_rank']
        state['probe_adam_per_rank'] = raw['probe_adam_per_rank']
    state.update(complete_outer=complete, latest_complete_checkpoint=complete,
        actual_native_state=summary['native'], stage=(raw.get('stage') or {}).get('stage'),
        last_actual_observation=str(raw_path),
        last_actual_observation_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        last_actual_utc=summary['observed_utc'],
        initial_five_accepted=bool(summary['initial_five_accepted']),
        initial_outers_verified=summary['new_verified_outers'],
        updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    qa = raw.get('qa5')
    if qa is not None:
        if (qa['run_id'] != state['run_id'] or qa['qa_count'] != 512
                or qa['evaluation_contract']['kind'] != 'unified_context4096_bf16_stream_action_v1'
                or qa['evaluation_contract']['compute_dtype'] != 'bfloat16'
                or qa['actual_worker_exit_codes'] != {str(i): 0 for i in range(8)}):
            raise ValueError('New QA5 needs the actual new run/contract/512/eight successful workers')
        state['qa5'] = qa
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
    compact = dict(summary, run_id=state['run_id'], base=state['base'],
                   last_original_sha256=state['last_actual_observation_sha256'])
    (state_path.parent / 'actorhidden-period20-latest-compact.json').write_text(
        json.dumps(compact, ensure_ascii=False, indent=2) + '\n')
    return dict(job=state['current_job_id'], complete_outer=complete,
                native=state['actual_native_state'], stage=state['stage'],
                initial_five_accepted=state['initial_five_accepted'], qa5=state.get('qa5'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', required=True)
    parser.add_argument('--directory', required=True)
    args = parser.parse_args()
    print(json.dumps(ingest(args.state, args.directory), ensure_ascii=False))
