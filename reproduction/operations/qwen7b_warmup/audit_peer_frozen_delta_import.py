"""Independent small CPU terminal join audit of the explicit frozen raw import.

Receiver/ingester already checked payload SHA, own strict EM, and causal seed
provenance. This rechecks descriptors and independently joins ordered own G5
identities/labels and exclusions. No tensor, payload row, or final-role source
is opened; real per-question records stay remote.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

STATUS='FROZEN_DELTA_RECEIVER_AND_ELIGIBLE_RAW_G10_INDEPENDENTLY_ACCEPTED'



def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def read(p):return json.loads(Path(p).read_bytes())
def fp(p):
    p=Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def checked(item):
    p=Path(item['path'])
    if p.is_symlink() or not p.is_file() or any(p.is_symlink() for p in p.parents):raise ValueError('Real regular artifact required')
    if fp(p)!={k:item[k] for k in ('path','bytes','sha256')}:raise ValueError('Descriptor SHA/bytes differs')
    return p

def own(g,n):
    if len(g['trajectory_ids'])!=n or len(set(g['trajectory_ids']))!=n or len(g['labels'])!=n or any(type(y)is not int or y not in(0,1) for y in g['labels']):raise ValueError('Own binary trajectory membership differs')


def run(ingest,selection,selection_sha,record_sha):
    ingest=Path(ingest);selection=Path(selection);s=read(selection);RECORD_SHA=record_sha
    if fp(selection)['sha256']!=selection_sha:raise ValueError('Exact published frozen selection file SHA required')
    count=s['questions'];trajectories=s['trajectories'];batches=s['batches'];archives=(batches+15)//16
    if type(count) is not int or count<1 or trajectories!=count*5 or batches!=len(s['selected_batches']):raise ValueError('Typed frozen ownG5 budget required')
    if (s['batch_records_sha256']!=RECORD_SHA or hashlib.sha256(json.dumps(s['selected_batches'],sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()!=RECORD_SHA
            or sum(v['questions'] for v in s['selected_batches'])!=count):raise ValueError('Frozen exact selection differs')
    done=read(ingest/'complete.json');c=read(checked(done['config']));receiver=Path(c['receiver']);r=read(checked(c['receiver_complete']));catalog=read(checked(c['receiver_catalog']))
    if (done['status']!='RECEIVED_NEWSEED5_PLUS_CANONICAL_OLD5_RAW_G10_STRICT_EM_CAUSAL_SEEDS_ACCEPTED'
            or r['status']!='PEER_NEWSEED_RAW_ALL_ARCHIVES_AND_FILES_FRESH_SHA256_VERIFIED'
            or r['questions']!=count or r['trajectories']!=trajectories or r['groups']!=archives
            or not r['all_archive_SHA256'] or not r['fresh_all_file_SHA256'] or done['strict_EM_mismatches']!=0
            or done['actual_same_question_old_new_seed_collisions']!=0 or done['own_strict_EM_recomputed']!=done['trajectories']):raise ValueError('Actual target/strictEM/causal-seed acceptance required')
    root=receiver/'raw'/receiver.name;sourceplan=read(root/'group-00000/provenance/plan.json')
    expected_overlap=dict(batches=0,qids=0,question_hashes=0,trajectory_ids=0)
    if s.get('training_text_hash_policy')is not None:
        if s['training_text_hash_policy']!='retain canonical distinct qids/authentic trajectory IDs; train-only repeated text hashes are reported; evaluation qid/hash exclusion remains hard':raise ValueError('Unknown canonical training text policy')
        expected_overlap['question_hashes']=s['prior_identity_overlap']['question_hashes']
        counts=sourceplan['canonical_train_text_hash_repeats']
        if counts['prior_hash_overlap_questions']!=s['prior_identity_overlap']['question_hashes']or counts['internal_duplicate_hash_questions']!=s['within_selected_training_hash_duplicates']:raise ValueError('Actual retained canonical train text-repeat counts differ')
    if sourceplan['subset_batch_records_sha256']!=RECORD_SHA or sourceplan['prior_identity_overlap']!=expected_overlap:raise ValueError('Fresh subset provenance differs')
    if [x['input'] for x in sourceplan['selected_batches']]!=[x['input'] for x in s['selected_batches']]:raise ValueError('Subset selected input list differs')
    bank=read(checked(c['original_bank']));old={g['qid']:g for g in bank['train']};new={};newtids=set();domains={}
    input_bindings={x['input']:x for x in s['selected_batches']};actual_inputs=set()
    for archive in catalog['groups']:
        name=archive['group'];prefix=root/name;cohort=read(prefix/'cohort.json')
        for path in (prefix/'source/inputs').glob('shard-*/batch*.json'):
            rel=str(path.relative_to(prefix/'source'));actual_inputs.add(rel);item=input_bindings[rel]
            if sha(path)!=item['input_sha256']:raise ValueError('Received canonical input SHA differs')
        for g in cohort['groups']:
            # Identity exclusions precede any own-label validation below.
            if g['qid'] in new or newtids.intersection(g['trajectory_ids']):raise ValueError('New own identity repeats')
            new[g['qid']]=g;newtids.update(g['trajectory_ids']);domain=json.loads(g['qid'])[0];domains[domain]=domains.get(domain,0)+1
    if actual_inputs!=set(input_bindings) or len(new)!=count or len(newtids)!=trajectories or domains!=s['domain_counts']:raise ValueError('Exact frozen question/domain/ownID coverage differs')
    excluded_qid={};excluded_hash=set();rejected={role:0 for role in c['exclusions']}
    for role,scope in c['exclusions'].items():
        for g in scope['groups']:
            excluded_qid.setdefault(g['qid'],[]).append(role);excluded_hash.add(g['question_hash'])
    expected=set();hash_excluded=0
    for q,g in new.items():
        if q in excluded_qid:
            for role in excluded_qid[q]:rejected[role]+=1
        elif g['question_hash'] in excluded_hash:hash_excluded+=1
        else:expected.add(q)
    groups=read(checked(done['outputs']['groups']))['train'];seen=set();tids=set();hist=[0]*11;oldhist=[0]*6;newhist=[0]*6;outdomains={}
    for g in groups:
        own(g,10);q=g['qid']
        if q not in expected or q in seen or tids.intersection(g['trajectory_ids']):raise ValueError('Eligible rawG10 repeated/foreign/heldout identity')
        a,b=old[q],new[q];own(a,5);own(b,5)
        if (g['question_hash']!=a['question_hash'] or g['question_hash']!=b['question_hash'] or g['trajectory_ids']!=a['trajectory_ids']+b['trajectory_ids'] or g['labels']!=a['labels']+b['labels']):raise ValueError('Old5+new5 authentic ordered ownEM join differs')
        for event,start in zip(g['real_G5_events'],(0,5)):
            if event['trajectory_ids']!=g['trajectory_ids'][start:start+5] or event['labels']!=g['labels'][start:start+5]:raise ValueError('Real event identity differs')
        seen.add(q);tids.update(g['trajectory_ids']);hist[sum(g['labels'])]+=1;oldhist[sum(g['labels'][:5])]+=1;newhist[sum(g['labels'][5:])]+=1
        domain=json.loads(q)[0];outdomains[domain]=outdomains.get(domain,0)+1
    if seen!=expected or len(groups)!=done['questions'] or len(tids)!=done['trajectories'] or (hist,oldhist,newhist)!=(done['histogram_K0_to10'],done['old5_histogram_K0_to5'],done['new5_histogram_K0_to5']):raise ValueError('Actual ownG10 coverage/histogram differs')
    overlap=read(checked(done['outputs']['overlap']))
    if overlap['excluded_by_qid']!=rejected or overlap['additional_excluded_by_question_hash']!=hash_excluded:raise ValueError('Independent identity exclusion count differs')
    if fp(ingest/'config.json')!=done['config']:raise ValueError('Config changed during terminal audit')
    resealed_expected=sum(bool(x['audit'].get('metadata_resealed_attestation'))for x in s['selected_batches']);events=read(checked(done['outputs']['source-events']));actual_resealed=[b for event in events['events']if event['event']=='newseed20261021'for b in event['batches']if b.get('metadata_resealed_attestation')]
    if len(actual_resealed)!=resealed_expected or any(b.get('original_receipt_chain_equal_current')is not False for b in actual_resealed):raise ValueError('Explicit one-batch resealed-metadata provenance differs')
    result=dict(status=STATUS,retained_canonical_training_hash_overlap=expected_overlap['question_hashes'],metadata_resealed_attestation_batches=resealed_expected,original_metadata_receipt_chain_not_silently_rewritten=True,selection=fp(selection),ingest_complete=fp(ingest/'complete.json'),receiver_complete=c['receiver_complete'],groups=done['outputs']['groups'],source_events=done['outputs']['source-events'],subset_record_sha256=RECORD_SHA,
        received_questions=count,received_newseed5_trajectories=trajectories,received_batches=batches,archive_groups=archives,received_domains=domains,
        eligible_training_questions=len(groups),eligible_authentic_G10_trajectories=len(tids),eligible_domains=outdomains,excluded_by_qid=rejected,additional_excluded_by_question_hash=hash_excluded,
        histogram_K0_to10=hist,old5_histogram_K0_to5=oldhist,new5_histogram_K0_to5=newhist,
        own_trajectory_IDs_and_ordered_old5_new5_strict_EM_join_verified=True,tokenizer_strict_EM_and_seed_audit_inherited_from_actual_ingester=True,
        final_identity_source_refs_followed=False,final_rollout_or_prediction_opened=False,model_tensor_loaded=False,actor_updates=0,probe_updates=0,active_training_data_modified=False,Mac_payload_transit=False,epoch=time.time())
    output=ingest/'independent-final-audit.json'
    if output.exists():
        previous=read(output)
        for key in ('ingest_complete','receiver_complete','groups','source_events','selection'):
            if previous[key]!=result[key]:raise ValueError('Existing terminal acceptance binding changed')
        return previous
    temp=output.with_name(output.name+'.tmp');temp.write_text(json.dumps(result,sort_keys=True)+'\n');temp.replace(output);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--ingest',type=Path,required=True);p.add_argument('--selection',type=Path,required=True);p.add_argument('--selection-sha256',required=True);p.add_argument('--record-sha256',required=True);a=p.parse_args()
    print(json.dumps(run(a.ingest,a.selection,a.selection_sha256,a.record_sha256)),flush=True)
if __name__=='__main__':main()
