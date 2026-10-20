"""Export only the frozen audited122 delta; reuse the accepted raw exporter.

The producer/source is read-only. No global discovery, generation, GPU, changed
old snapshot, or payload transport through the control machine is performed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from . import peer_newseed_export as e
from . import peer_newseed_delta as d

RECORD_SHA = '673b8cb6abe7108e1d03a98b518121dafcd1ec10b156ec78f37c5faacfb7cc21'
STATUS = 'FROZEN_122_AUDITED_DELTA_RAW_SNAPSHOT_FRESH_SHA_ACCEPTED'
BASE = Path(e.RECORDED)


def digest(value):
    return hashlib.sha256(e.canonical(value)).hexdigest()


def checked(ref):
    p=Path(ref['path']); e.regular(p.parent,p)
    if p.stat().st_size!=ref['bytes'] or e.sha(p)!=ref['sha256']:
        raise ValueError('Frozen source descriptor changed: '+str(p))
    return p


def validate_selection(value):
    if (value.get('status')!='FROZEN_METADATA_DELTA_SUBSET_NOT_EXPORTED' or value.get('subset')!='existing-audited'
            or value.get('source_base')!=str(BASE) or value.get('source_run_id')!=e.RUN_ID
            or value.get('seed')!=e.SEED or value.get('model_revision')!=e.REVISION
            or value.get('sampling_attempt_id')!=e.ATTEMPT or value.get('batches')!=122
            or value.get('questions')!=15616 or value.get('trajectories')!=78080
            or value.get('domain_counts')!={'nq':12300,'hotpotqa':3316}
            or value.get('batch_records_sha256')!=RECORD_SHA
            or digest(value.get('selected_batches'))!=RECORD_SHA):
        raise ValueError('Requires exactly the published audited122 delta; never discover newly sealed batches')
    values=value['selected_batches']
    if len(values)!=122 or len({x['input'] for x in values})!=122 or values!=sorted(values,key=lambda x:x['input']):
        raise ValueError('Exact distinct frozen batch order required')
    if any(x['audit']['status']!='EXISTING_INDEPENDENT_AUDIT_CURRENT_FSTAT_BOUND' for x in values):
        raise ValueError('Every selected batch needs its frozen existing independent proof')
    return value


def identity_guard(old,groups,seen):
    """No old244/batch or question/hash/authentic-five identity may recur."""
    for group in groups:
        qid,qhash=group['qid'],group['question_hash']; tids=group['trajectory_ids']
        if qid in old['qids'] or qhash in old['hashes'] or old['tids'].intersection(tids):
            raise ValueError('Subset overlaps the accepted imported244 identities')
        if qid in seen['qids'] or seen['tids'].intersection(tids):
            raise ValueError('Subset duplicate own question/trajectory identity')
        seen['qids'].add(qid);seen['hashes'].add(qhash);seen['tids'].update(tids)


def freeze(base,selection_path,root):
    base=Path(base); root=Path(root); selection_path=Path(selection_path)
    selection=validate_selection(e.read(selection_path)); selref=e.fp(selection_path)
    if base!=BASE or base.resolve()!=base:
        raise ValueError('Canonical approved physical source only')
    config=checked(selection['source_config']); manifest=checked(selection['input_manifest'])
    if config!=base/'config.json' or manifest!=base/'inputs/manifest.json':
        raise ValueError('Canonical root source binding differs')
    d.contract(e.read(config)); declared=e.read(manifest)
    if declared['questions']!=169028 or len(declared['batches'])!=1322:
        raise ValueError('Original full input manifest differs')
    previous=checked(selection['imported_snapshot_complete']); prior=e.read(previous)
    pp=checked(prior['plan']); membership=checked(prior['canonical_membership'])
    oldplan=e.read(pp); oldgroups=e.read(membership)['groups']
    if prior['questions']!=31232 or prior['frozen_sealed_batches']!=244 or len(oldgroups)!=31232:
        raise ValueError('Actual imported244 baseline differs')
    oldinputs={x['input'] for x in oldplan['selected_batches']}
    old=dict(qids={g['qid'] for g in oldgroups},hashes={g['question_hash'] for g in oldgroups},tids={t for g in oldgroups for t in g['trajectory_ids']})
    seen=dict(qids=set(),hashes=set(),tids=set()); selected=[]; domains={}
    for index,item in enumerate(selection['selected_batches']):
        if item['input'] in oldinputs:raise ValueError('Imported batch repeated in delta')
        inp=checked(item['input_ref']); seal=e.regular(base,base/item['seal'])
        if inp!=base/item['input'] or e.sha(seal)!=item['seal_sha256'] or e.sha(inp)!=item['input_sha256']:
            raise ValueError('Frozen input/seal differs')
        cfg=checked(item['config']);d.contract(e.read(cfg))
        auditfile=checked(item['audit']['proof']);proof=e.read(auditfile)
        fresh=d.cached_audit(base,item['input'],selection['source_config']['sha256'],item['input_sha256'],item['questions'])
        if (fresh['status']!='EXISTING_INDEPENDENT_AUDIT_CURRENT_FSTAT_BOUND' or fresh['proof']['sha256']!=item['audit']['proof']['sha256']
                or fresh['own_trajectory_ids_sha256']!=item['expected_own_G5_IDs_sha256']):
            raise ValueError('Frozen independent audit no longer matches every actual source file fstat')
        rows=e.read(inp)['rows']; own=[];qids=[];hashes=[]; counts={}
        for row in rows:
            if row['split']!='train':raise ValueError('Nontraining source question')
            tids=[e.RUN_ID+'-outer000001-c%04d-g%d'%(row['candidate_ordinal'],j) for j in range(5)]
            own.append(dict(qid=row['qid'],question_hash=row['question_hash'],trajectory_ids=tids))
            qids.append(row['qid']);hashes.append(row['question_hash']);domain=json.loads(row['qid'])[0]
            counts[domain]=counts.get(domain,0)+1;domains[domain]=domains.get(domain,0)+1
        if (len(rows)!=item['questions'] or digest(qids)!=item['ordered_qids_sha256']
                or digest(hashes)!=item['ordered_question_hashes_sha256'] or counts!=item['domain_counts']
                or digest([t for g in own for t in g['trajectory_ids']])!=item['expected_own_G5_IDs_sha256']):
            raise ValueError('Frozen label-blind actual input identities differ')
        identity_guard(old,own,seen)
        selected.append(dict(input=item['input'],input_sha256=item['input_sha256'],input_stat=e.fs(inp),seal=item['seal'],seal_sha256=item['seal_sha256'],seal_stat=e.fs(seal),questions=item['questions'],trajectories=item['trajectories']))
        e.atomic(root/'subset-preflight-progress.json',dict(stage='FROZEN_SUBSET_INPUT_AND_CURRENT_PROOF_ADVANCING',batches=index+1,questions=len(seen['qids']),trajectories=len(seen['tids']),epoch=time.time()))
    if len(seen['qids'])!=15616 or len(seen['tids'])!=78080 or domains!=selection['domain_counts']:
        raise ValueError('Frozen subset total/domain identity differs')
    # The existing engine archives only this explicit list. No unselected batch
    # seal or payload is discovered; out-of-subset is not called "unsealed".
    return dict(source_base=str(base),source_node='h100-10',source_run_id=e.RUN_ID,seed=e.SEED,revision=e.REVISION,
        source_config=e.fp(config),input_manifest=e.fp(manifest),selected_batches=selected,unsealed_inputs_excluded=[],
        source_input_batches=1322,source_all_inputs_sealed=False,questions=15616,trajectories=78080,
        cohort_seal_digest=digest(selected),discovery_verified_payloads=False,
        selection_rule='EXACT published audited122 delta; no global discovery or label/probe selection',
        subset_selection=selref,subset_batch_records_sha256=RECORD_SHA,subset_exporter_sha256=e.sha(__file__),
        previous_imported_complete=e.fp(previous),previous_imported_membership=e.fp(membership),
        out_of_subset_not_inspected=True,old244_identity_overlap=dict(batches=0,qids=0,question_hashes=0,trajectory_ids=0))


def run(base,output,snapshot,selection_path):
    root=Path(output)/snapshot;root.mkdir(parents=True,exist_ok=True)
    frozen=freeze(base,selection_path,root)
    if (root/'plan.json').exists():
        prior=e.read(root/'plan.json')
        for key in ('subset_selection','subset_batch_records_sha256','subset_exporter_sha256','previous_imported_complete','old244_identity_overlap'):
            if prior.get(key)!=frozen[key]:raise ValueError('Existing subset plan binding differs')
    e.immutable(root/'plan.json',dict(frozen,snapshot_id=snapshot,batches_per_archive=16,exporter_sha256=e.sha(e.__file__),auditor_sha256=e.AUDIT_SHA))
    old_discover=e.discover
    try:
        e.discover=lambda actual: frozen if Path(actual)==Path(base) else (_ for _ in ()).throw(ValueError('Source differs'))
        done=e.export(Path(base),Path(output),snapshot,batches_per_archive=16)
    finally:e.discover=old_discover
    if done['questions']!=15616 or done['trajectories']!=78080 or done['frozen_sealed_batches']!=122:
        raise ValueError('Exported subset totals differ')
    result=dict(status=STATUS,snapshot_id=snapshot,selection=e.fp(selection_path),subset_record_sha256=RECORD_SHA,
        source_complete=e.fp(root/'complete.json'),source_plan=e.fp(root/'plan.json'),questions=15616,trajectories=78080,
        batches=122,archives=len(done['groups']),archive_bytes=sum(g['archive_bytes'] for g in done['groups']),
        old244_identity_overlap=frozen['old244_identity_overlap'],all_archive_and_member_SHA_fresh=True,
        no_global_discovery=True,producer_source_modified=False,GPU_started=False,epoch=time.time())
    accepted=root/'subset-complete.json'
    if accepted.exists():
        previous=e.read(accepted)
        for key in ('selection','subset_record_sha256','source_complete','source_plan','questions','trajectories','batches','archives','archive_bytes'):
            if previous[key]!=result[key]:raise ValueError('Existing subset acceptance changed')
        return previous
    e.immutable(accepted,result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--selection',type=Path,required=True);p.add_argument('--snapshot-id',required=True)
    p.add_argument('--base',type=Path,default=BASE);p.add_argument('--output-root',type=Path,default=Path(e.OUTPUT));a=p.parse_args()
    if sys.platform=='darwin' or a.base!=BASE or a.output_root!=Path(e.OUTPUT):p.error('Remote canonical source/independent output only')
    print(json.dumps(run(a.base,a.output_root,a.snapshot_id,a.selection)),flush=True)

if __name__=='__main__':main()
