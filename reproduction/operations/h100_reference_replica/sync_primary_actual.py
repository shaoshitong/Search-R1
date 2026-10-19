"""Root-only local merge of an existing observation, never SSH or training actions.

Keep historical alias fields consistent with the authoritative current counters.
Reject stale/wrong-run inputs without overwriting primary state. The read-only
observer does not call this module or write the primary state itself.
"""
import datetime
import hashlib
import json
import os
from pathlib import Path
import uuid

from .follow_h10011_ab_normal_boundary import ROOT


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def observation_time(value):
    if '-' in value:
        return datetime.datetime.fromisoformat(value.replace('Z', '+00:00'))
    return datetime.datetime.strptime(value, '%Y%m%dT%H%M%S%fZ').replace(tzinfo=datetime.timezone.utc)


def atomic(path, value):
    temporary = path.with_name(path.name + '.partial-' + uuid.uuid4().hex)
    with temporary.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def main():
    state_path = ROOT / 'work/search-g1-prefix5-ab200-state.json'
    compact = read(ROOT / 'work/prefix5-ab-latest-compact.json')
    source = Path(compact['original'])
    if source.parent != ROOT / 'work' or sha(source) != compact['original_sha256']:
        raise ValueError('Existing local original observation identity differs')
    original, state = read(source), read(state_path)
    if state.get('last_actual_utc') and observation_time(compact['observed_utc']) < observation_time(state['last_actual_utc']):
        print(json.dumps(dict(status='STALE_COMPACT_PRIMARY_RETAINED', reason='Older actual observation time', training_modified=False)))
        return
    new_observation = compact['original_sha256'] != state.get('last_actual_observation_sha256')
    if set(original) != {'A', 'B'}:
        raise ValueError('Both actual arms are required')
    for arm, actual in original.items():
        expected, native = state['slots'][arm], actual['native']
        if (actual['base'] != expected['base'] or native['run_id'] != expected['run_id']
                or native['config_sha256'] != expected['config_sha256']):
            raise ValueError('Original observation belongs to a different execution')
        if not 0 <= actual['complete_outer'] <= 200:
            raise ValueError('Actual outer lies outside the authorized200/no201 budget')
        if actual['complete_outer'] < expected['complete_outer']:
            print(json.dumps(dict(status='STALE_COMPACT_PRIMARY_RETAINED', arm=arm,
                incoming=actual['complete_outer'], current=expected['complete_outer'], training_modified=False)))
            return
    advanced = any(v['complete_outer'] > state['slots'][a]['complete_outer'] for a, v in original.items())
    facts = {}
    for arm, actual in original.items():
        slot = state['slots'][arm]; facts[arm] = actual['facts']
        slot.update(complete_outer=actual['complete_outer'], native=actual['native'],
            last_actual_observation=str(source), last_actual_observation_sha256=compact['original_sha256'],
            current_namespace_qa=actual['qa'])
        for name in ('actor_adam', 'actor_actual_adam_per_rank', 'actor_adam_per_rank'):
            slot[name] = actual['actor_adam']
        for name in ('probe_adam', 'probe_actual_adam_per_rank', 'probe_adam_per_rank'):
            slot[name] = actual['probe_adam']
        for name in ('stage', 'current_stage', 'actual_stage'):
            slot[name] = actual['stage']
        for name in ('current_score_adoption', 'latest_actual_score_adoption'):
            slot[name] = actual.get('current_score_adoption')
        slot.setdefault('qa', {}).update(actual['qa'])
        state['latest_restore_candidates'][arm] = actual['complete_outer']
        if actual['facts']:
            last = actual['facts'][-1]; metrics = last['selection_metrics']
            if last['outer'] != actual['complete_outer']:
                raise ValueError('Latest selected metrics are not from the actual complete outer')
            slot.update(latest_selected_outer=last['outer'], latest_selected_mixing=metrics,
                last_selected_metrics=metrics, last_selected_mixed_fraction=metrics['mixed_fraction'],
                last_selected_mixed_count=metrics['mixed'])
    counts = {a: v['complete_outer'] for a, v in original.items()}
    small = ROOT / ('work/prefix5-ab-a%d-b%d-actual-small-facts-%s.json' %
        (counts['A'], counts['B'], compact['observed_utc']))
    atomic(small, dict(original=str(source), original_sha256=compact['original_sha256'],
        facts=facts, training_modified=False))
    state.update(last_actual_observation=str(source), last_actual_observation_sha256=compact['original_sha256'],
        last_actual_utc=compact['observed_utc'], latest_completed_small_facts=str(small),
        latest_completed_small_facts_sha256=sha(small), updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    if advanced or new_observation:
        live = all(v['native'].get('leader_same_birth') is True and not v.get('cleanup') for v in original.values())
        state['current_goal_turn_classification'] = dict(
            classification='progress' if advanced or not live else 'verified_wait',
            reason='New original rank journals/current native state merged; aliases synchronized; no training action',
            source=str(source), terminal_attention=compact.get('terminal_attention', []))
    atomic(state_path, state)
    print(json.dumps(dict(status='ORIGINAL_LOCAL_OBSERVATION_MERGED', complete_outer=counts,
        actor_adam={a:v['actor_adam'] for a,v in original.items()},
        probe_adam={a:v['probe_adam'] for a,v in original.items()},
        terminal_attention=compact.get('terminal_attention', []), source=str(source), training_modified=False)))


if __name__ == '__main__':
    main()
