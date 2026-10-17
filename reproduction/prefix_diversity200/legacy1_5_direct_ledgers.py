"""Read only SHA-pinned original joint1..5 and small eight-rank actor/probe ledgers.

Reuses watch_actual.EXTRA's actual write count/counter/receipt checks, with the
requested scheduler boundary and explicit outer1 fresh-init treatment. Never
imports torch, reads weights, polls GPUs or rewrites a training artifact.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

PARENT = Path('/mnt/data/shaoshitong')
PREFIX = 'search-g1-prefix-diversity200-20261016-f3ee1435e501'
RUN = 'f3ee1435e5014ba1969387c91878893a'
JOINTS = {
    1: 'c0e306c030a27f847db251a8c57da56d619817e4a6320918d90bc4b17f3e5182',
    2: 'e7d3e32a72725888dd72d9d8ad3070b3f7dc04ffeede820606c2df48042bca73',
    3: 'a4aef5933ac1e71e0f0089d306a996560e1139cb3cd54d714b1af5e09a9c2b2e',
    4: 'ef4abfa02c9cb95db0d49ac53bdbb89f0a5a4e4fca42ff084a0179150031adad',
    5: 'c9e6cc2af7a5de8d10d297f9fefa6bdcc2a07b048c1584320ad519ae149564b0',
}
CACHE = {}


def need(condition, message):
    if not condition: raise ValueError(message)


def small(path):
    path = Path(path)
    need(path.is_absolute() and '..' not in path.parts and path.is_relative_to(PARENT), 'Foreign small artifact path')
    relative = path.relative_to(PARENT)
    need(relative.parts[0].startswith(PREFIX) and path.suffix in ('.json','.jsonl')
         and path.resolve() == path and not path.is_symlink(), 'Outside the retained run or redirected artifact')
    if str(path) in CACHE: return CACHE[str(path)]
    before = path.stat(); need(0 < before.st_size <= 4*1024**2, 'Refuse non-small JSON/JSONL')
    raw = path.read_bytes(); after = path.stat()
    fields = lambda s: (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
    need(fields(before) == fields(after), 'Original ledger changed during read')
    proof = dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),
                 inode=before.st_ino,mtime_ns=before.st_mtime_ns,independent_full_small_file_sha=True)
    CACHE[str(path)] = raw, proof
    return raw, proof


def read(path):
    raw, proof = small(path)
    return json.loads(raw), proof


def ledger(path):
    raw, proof = small(path)
    return [json.loads(line) for line in raw.splitlines()], proof


def verify(outer, candidates):
    expected = JOINTS[outer]; matches=[]; inspected=[]
    for base in candidates:
        path=base/'outers'/('outer-%06d'%outer)/'complete.json'
        if not path.is_file(): continue
        value, proof=read(path); inspected.append(dict(path=str(path),sha256=proof['sha256']))
        if proof['sha256']==expected: matches.append((path,value,proof))
    need(matches, 'Missing exact SHA-pinned original joint%d'%outer)
    path,joint,joint_file=matches[0];actor=joint['actor']
    need(joint['run_id']==actor['run_id']==joint['probe']['run_id']==RUN and joint['outer']==outer
         and actor['completed_outer']==joint['probe']['completed_outer']==outer
         and actor['actual_adam_steps_per_rank']==joint['probe']['actual_adam_steps_per_rank']==[20*outer]*8
         and actor['scheduler_completed_outer']==outer, 'Foreign joint boundary')
    # The actor may have completed in an earlier base before a failed probe
    # phase; its ledgers still come from rank_receipts below. The actual
    # SHA-matched joint root owns the later successful probe-update logs.
    actor_original_rollout_root=Path(actor['original_rollout_path']).parent.parent
    original_root=path.parent
    original_joint,original_proof=read(original_root/'complete.json')
    need(original_proof['sha256']==expected and original_joint==joint, 'Original rollout root has a different joint')
    need(len(actor['rank_receipts'])==8, 'Missing actor rank receipt')
    ar=[];pr=[];counter=list(range(20*(outer-1)+1,20*outer+1))
    for rank in range(8):
        receipt=actor['rank_receipts'][rank];rows,proof=ledger(receipt['ledger_path'])
        writes=[row for row in rows if row['event']=='ACTUAL_ADAM_WRITE']
        restores=[row for row in rows if row['event']=='ACTUAL_RESTORE']
        need(proof['sha256']==receipt['ledger_sha256'] and receipt['rank']==rank and receipt['outer']==outer,
             'Original actor ledger SHA/rank/outer differs')
        need(len(writes)==20 and [row['actual_adam_steps'] for row in writes]==counter
             and all(row['rank']==rank and row['outer']==outer and row['opportunity']==step
                     and row['writes_this_attempt']==step+1 for step,row in enumerate(writes)),
             'Actor did not perform exactly20 contiguous rank-local writes')
        if outer==1 and not restores:
            mode='FRESH_OUTER1_NO_ACTUAL_RESTORE_RECORD'
            restore=None
        else:
            need(len(restores)==1, 'Expected one original actual restore receipt')
            restore=restores[0]
            need(restore['rank']==rank and restore['outer']==outer
                 and restore['restored_completed_outer']==outer-1
                 and restore['restored_scheduler']==outer-1
                 and restore['actual_adam_before']==20*(outer-1), 'Actor restore/scheduler/Adam boundary differs')
            mode='FRESH_OUTER1_ZERO_BOUNDARY_EVENT' if outer==1 else 'RESTORED_PREVIOUS_COMPLETE_OUTER'
            if outer==1:
                need(restore['actor_version']==RUN+'-official0', 'Outer1 zero-boundary event is not official fresh actor')
        ar.append(dict(rank=rank,**proof,writes=20,first_counter=counter[0],last_counter=counter[-1],
                       initialization_or_restore=mode,restore=restore,first=writes[0],last=writes[-1]))
        rows,proof=ledger(original_root/'probe-update'/('rank-%d-updates.jsonl'%rank))
        need(len(rows)==20 and [row['actual_adam_steps'] for row in rows]==counter
             and all(row['rank']==rank and row['outer']==outer and row['step']==step
                     for step,row in enumerate(rows,1)), 'Probe did not record20 contiguous synchronous writes')
        pr.append(dict(rank=rank,**proof,writes=20,first_counter=counter[0],last_counter=counter[-1],
                       first=rows[0],last=rows[-1]))
    return dict(outer=outer,status='ORIGINAL_8RANK_SMALL_LEDGER_CHECKS_PASS',
        independent_direct_8rank_ledger_facts=True,joint_file=joint_file,original_joint_file=original_proof,
        matching_joint_paths=[str(item[0]) for item in matches],inspected_small_joint_paths=inspected,
        original_root=str(original_root),actor_original_rollout_root=str(actor_original_rollout_root),actor=ar,probe=pr,
        actor_actual_steps_per_rank=[20*outer]*8,probe_actual_steps_per_rank=[20*outer]*8,
        scheduler_completed_outer=outer,
        outer1_fresh_scope='Outer1 is explicitly fresh; ACTUAL_RESTORE may name a zero-boundary initialization event. No preceding trained checkpoint is inferred.' if outer==1 else None)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outer',type=int,choices=tuple(JOINTS),action='append')
    args=parser.parse_args();requested=args.outer or list(JOINTS)
    need(len(requested)==len(set(requested)), 'Do not duplicate an outer in one small-file read')
    started=time.time()
    # Directory names only; the sole file search is exactly these5 fixed joint
    # paths per retained base. Never recurse through logs/checkpoints.
    bases=sorted(p for p in PARENT.glob(PREFIX+'*') if p.is_dir() and not p.is_symlink())
    facts=[]
    for outer in requested:
        try: facts.append(verify(outer,bases))
        except (OSError,ValueError,KeyError,TypeError,json.JSONDecodeError) as error:
            facts.append(dict(outer=outer,status='MISSING_OR_INVALID_ORIGINAL_SMALL_LEDGER',
                              independent_direct_8rank_ledger_facts=False,error_type=type(error).__name__,error=str(error)))
    passed=all(row['independent_direct_8rank_ledger_facts'] for row in facts)
    print(json.dumps(dict(status='REQUESTED_LEGACY_DIRECT_SMALL_LEDGER_PASS' if passed else 'REQUESTED_LEGACY_DIRECT_SMALL_LEDGER_INCOMPLETE',
        run_id=RUN,started_unix=started,finished_unix=time.time(),searched_retained_bases=[str(p)for p in bases],
        requested_outers=requested,expected_joint_sha256={n:JOINTS[n]for n in requested},facts=facts,
        small_file_count=len(CACHE),small_bytes_read=sum(len(v[0])for v in CACHE.values()),
        validation_contract='watch_actual.EXTRA exact20/counter/receipt checks plus explicit scheduler and outer1 fresh treatment',
        weights_or_tensors_read=False,gpu_calls=False,training_modified=False,legacy_series_modified=False,
        previous_cpu5_or_cpu50_audit_replayed=False),sort_keys=True,allow_nan=False))


if __name__=='__main__':main()
