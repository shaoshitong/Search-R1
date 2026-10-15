"""Bounded read-only fresh-epoch acceptance collector; explicit invocation only.

Reads existing small receipts/ledgers and stats payloads. Never hashes/loads a
model, hashes rollout payloads, writes remotely, or changes job/state/GPU work.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sys

REPO = '/Users/shaoshitong/Documents/Search-G1/qprobe-prefix20-online1000-h1008-20261016'
DEFAULT_BASE = '/mnt/data/shaoshitong/search-g1-prefix20-online1000-20261016-ca1c60f'
RUNTIME = 'ca1c60f49a29c30ca299c1d6073ee9a4bb86655b'

REMOTE = r'''
from pathlib import Path
import datetime,hashlib,json
A=ARGS
B=Path(A['base']);FIRST=A['first_epoch'];LAST=A['last_epoch'];NEXT=A['expected_next_epoch']
MAX_SMALL=16*1024*1024
proofs=[]
def small(path,jsonl=False):
    p=Path(path);before=p.stat()
    if before.st_size>MAX_SMALL:raise ValueError('Refuse oversized bounded receipt: '+str(p))
    raw=p.read_bytes();after=p.stat()
    if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns):
        raise ValueError('Receipt changed during read: '+str(p))
    proof=dict(path=str(p),bytes=len(raw),independent_small_file_full_sha256=hashlib.sha256(raw).hexdigest())
    proofs.append(proof)
    return ([json.loads(line) for line in raw.decode().splitlines() if line] if jsonl else json.loads(raw)),proof

def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()

def asset(record,role,rank):
    p=Path(record['path']);st=p.stat()
    if st.st_size!=record['bytes']:raise ValueError('Payload size differs from receipt: '+str(p))
    return dict(role=role,rank=rank,path=str(p),bytes=st.st_size,count=record['count'],
                producer_receipt_sha256=record['sha256'],independent_payload_full_sha_performed=False,
                sha_semantics='Producer receipt hash; this audit only stats the payload, never hashes or reads it.')

config,config_proof=small(B/'preparation/launch-config.json')
if config['source_commit']!=A['runtime_source_commit']:raise ValueError('Frozen runtime identity differs')
# Stream bounded existing feature-index metadata. Do not read feature arrays,
# question text, token arrays, model files or raw rollouts.
test_index=Path(config['cached_features'])/'index.jsonl'
if test_index.stat().st_size>128*1024*1024:raise ValueError('Refuse oversized identity metadata index')
fixed_qids=set();fixed_rows=0
with test_index.open() as stream:
    for line in stream:
        if len(line)>1024*1024:raise ValueError('Oversized identity metadata line')
        row=json.loads(line)
        if row['split']=='test':fixed_qids.add(str(row['qid']));fixed_rows+=1
if len(fixed_qids)!=512 or fixed_rows!=2560:raise ValueError('Fixed diagnostic membership differs')
fixed_qid_proof=dict(index_path=str(test_index),index_bytes=test_index.stat().st_size,
                     diagnostic_questions=512,diagnostic_trajectories=2560,
                     sorted_qid_set_sha256=hashlib.sha256(canonical(sorted(fixed_qids))).hexdigest(),
                     full_index_independent_sha_performed=False,metadata_streamed_only=True)

previous_selection,_=small(B/'epochs/epoch-0001/selection.json')
previous_qids={str(row['qid']) for row in previous_selection['rows']}
historical_metrics,historical_metrics_proof=small(B/'train/epochs/epoch-0001/metrics.json')
historical_complete,historical_complete_proof=small(B/'train/epochs/epoch-0001/complete.json')
# Console stores actual stage times. Limit the read and return only phase JSON.
console=[]
log=B/'logs'/('native-%d.log'%A['job_id'])
if log.exists():
    if log.stat().st_size>8*1024*1024:raise ValueError('Refuse oversized console read')
    for line in log.read_text(errors='replace').splitlines():
        if 'PREFIX20_PIPELINE=' not in line:continue
        try:value=json.loads(line.split('PREFIX20_PIPELINE=',1)[1])
        except ValueError:continue
        epoch=value.get('epoch')
        if isinstance(epoch,int) and FIRST<=epoch<=NEXT:console.append(value)

all_qids=set();all_normalized=set();epochs=[];pairwise=[];test_manifests=set()
for epoch in range(FIRST,LAST+1):
    E=B/'epochs'/('epoch-%04d'%epoch);T=B/'train/epochs'/('epoch-%04d'%epoch)
    selection,sp=small(E/'selection.json');questions,qp=small(E/'questions.json')
    rollout,rp=small(E/'complete.json');complete,cp=small(T/'complete.json');metrics,mp=small(T/'metrics.json')
    pool,pp=small(E/'pool-state-at-head-completion.json')
    if any(record['epoch']!=epoch for record in (selection,questions,rollout,complete,metrics)):
        raise ValueError('Epoch identity mismatch')
    if rollout['run_id']!=config['run_id'] or rollout.get('source_commit')!=A['runtime_source_commit']:
        raise ValueError('Rollout producer identity mismatch')
    if rollout['questions']!=5120 or rollout['trajectories']!=25600:raise ValueError('Incomplete G5 rollout')
    if questions['selection_sha256']!=sp['independent_small_file_full_sha256'] or rollout['selection_sha256']!=sp['independent_small_file_full_sha256']:
        raise ValueError('Question/rollout selection binding differs')
    if rollout['questions_sha256']!=qp['independent_small_file_full_sha256']:
        raise ValueError('Question bundle hash differs from rollout')
    rows=selection['rows'];qids={str(row['qid'])for row in rows};normalized={row['normalized_question_hash']for row in rows}
    if len(rows)!=5120 or len(qids)!=5120 or len(normalized)!=5120:raise ValueError('Repeated selected question or content')
    if [str(row['qid'])for row in questions['rows']]!=[str(row['qid'])for row in rows]:
        raise ValueError('Saved original-question order differs')
    if selection['fixed_test_qids_sha256']!=fixed_qid_proof['sorted_qid_set_sha256']:
        raise ValueError('Diagnostic-exclusion QID binding differs')
    overlap=dict(epoch=epoch,prior_fresh_qid_overlap=len(qids&all_qids),
                 prior_fresh_normalized_content_overlap=len(normalized&all_normalized),
                 cached_epoch1_qid_overlap=len(qids&previous_qids),diagnostic_qid_overlap=len(qids&fixed_qids))
    if any(overlap[key]for key in overlap if key!='epoch'):raise ValueError('Unexpected selected-question overlap')
    pairwise.append(overlap);all_qids.update(qids);all_normalized.update(normalized)
    ranks=[];assets=[]
    for rank in range(8):
        rr,rrp=small(E/'rollout'/('rank-%d-complete.json'%rank));updates,up=small(T/('rank-%d-updates.jsonl'%rank),True)
        if(rr['rank'],rr['epoch'],rr['questions'],rr['trajectories'])!=(rank,epoch,640,3200):
            raise ValueError('Rollout rank boundary differs')
        before=(epoch-1)*100;after=epoch*100
        if [row['actual_adam_steps']for row in updates]!=list(range(before+1,after+1)):
            raise ValueError('Actual rank Adam ledger is not exactly100 successive writes')
        if any(row['epoch']!=epoch or row['local_trajectories']!=32 for row in updates):
            raise ValueError('Actual rank batch budget differs')
        ranks.append(dict(rank=rank,questions=640,trajectories=3200,strict_correct=rr['strict_correct_trajectories'],
             rollout_elapsed_seconds=rr['elapsed_seconds'],rollout_complete=rrp,update_ledger=up,
             actual_adam_before=before,actual_adam_after=after,actual_adam_increment=100,actual_update_records=100,
             first_update=updates[0],last_update=updates[-1]))
        assets.extend((asset(rr['prefix'],'reusable_compact_prefix20',rank),asset(rr['full'],'original_full_rollout',rank)))
    expected=[epoch*100]*8
    if complete['actual_adam_steps_per_rank']!=expected or metrics['actual_adam_steps_per_rank']!=expected:
        raise ValueError('Actual completed eight-rank Adam counts differ')
    if metrics['train_unique_trajectories']!=25600 or metrics['sampling']!='natural_no_replacement':
        raise ValueError('Fresh epoch is not one complete natural pass')
    if metrics['raw']['count']!=2560 or metrics['raw']['positive']!=559 or metrics['raw']['negative']!=2001:
        raise ValueError('Fixed diagnostic sample/label counts differ')
    if metrics['test_manifest_sha256']!=config['cached_features_manifest_sha256']:
        raise ValueError('Fixed diagnostic cache binding differs')
    test_manifests.add(metrics['test_manifest_sha256'])
    if pool['next_epoch']!=epoch+1:raise ValueError('Immutable pool/head boundary differs')
    phase=[record for record in console if record.get('epoch')==epoch]
    phasewall={}
    for left,right in zip(console,console[1:]):
        if left.get('epoch')==epoch and left['stage'].startswith('e%04d-'%epoch):
            phasewall[left['stage']]=right['checked_unix']-left['checked_unix']
    feature=[]
    for rank in range(8):
        f=E/'features'/('shard-%d-complete.json'%rank)
        if f.exists():
            record,proof=small(f);feature.append(dict(rank=rank,count=record['count'],total_tokens=record['total_tokens'],elapsed_seconds=record['elapsed_seconds'],receipt=proof))
    pruned=None
    if(E/'derived-cache-pruned.json').exists():pruned,_=small(E/'derived-cache-pruned.json')
    epochs.append(dict(epoch=epoch,questions=5120,unique_questions=5120,trajectories=25600,
         rank_facts=ranks,actual_adam_before_per_rank=[(epoch-1)*100]*8,
         actual_adam_after_per_rank=expected,actual_adam_increment_per_rank=[100]*8,rank_Adam_not_summed=True,
         selection=sp,questions_bundle=qp,rollout_complete=rp,train_complete=cp,metrics=metrics,metrics_proof=mp,
         retained_rollout_assets=assets,question_overlap_counts=overlap,
         fixed_diagnostic_normalized_text_sha256=selection['fixed_test_normalized_text_sha256'],
         pool_boundary=dict(proof=pp,seed=pool['seed'],cursor=pool['cursor'],cycle=pool['cycle'],next_epoch=pool['next_epoch'],consumed_questions=pool.get('consumed_questions')),
         checkpoint_receipt=dict(epoch=epoch,path=str(B/'train/latest.pt'),producer_checkpoint_sha256=complete['sha256'],
              bytes=complete['bytes'],immutable_saved=complete['immutable_saved'],independent_model_full_sha_performed=False,
              path_semantics='latest.pt is mutable and may now represent a later epoch; this SHA is the historical completed-epoch producer receipt.'),
         phase_records=phase,phase_wall_seconds_including_startup_polling_cleanup_publication=phasewall,
         training_loop_seconds=metrics['train_seconds'],fixed_diagnostic_loop_seconds=metrics['diagnostic_seconds'],
         encoder_rank_receipts=feature,derived_cache_pruned=pruned))

if len(test_manifests)!=1:raise ValueError('Diagnostic manifest differs across fresh epochs')
if len({row['fixed_diagnostic_normalized_text_sha256']for row in epochs})!=1:
    raise ValueError('Normalized diagnostic-exclusion binding differs across epochs')
stage,stage_proof=small(B/'logs/pipeline-status.json');latest_bundle,latest_bundle_proof=small(B/'train/latest-bundle.json')
next_progress={}
for rank in range(8):
    p=B/'epochs'/('epoch-%04d'%NEXT)/'rollout'/('rank-%d-progress.json'%rank)
    if p.exists():
        value,proof=small(p)
        next_progress[str(rank)]=dict(proof=proof,epoch=value['epoch'],stage=value['stage'],
             completed_trajectories=value['completed_trajectories'],generation_requests=value['generation_requests'],
             active_trajectories=value['active_trajectories'],elapsed_seconds=value['elapsed_seconds'])
    else:next_progress[str(rank)]=None
next_started=any(value and value['epoch']==NEXT and value['completed_trajectories']>0 for value in next_progress.values())
result=dict(role='FIVE_FRESH_EPOCH_ACTUAL_FACTS',observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    base=str(B),job_id=A['job_id'],run_id=config['run_id'],runtime_source_commit=A['runtime_source_commit'],
    first_epoch=FIRST,last_epoch=LAST,expected_next_epoch=NEXT,fresh_complete_epochs=len(epochs),
    fresh_questions_processed=5120*len(epochs),fresh_unique_selected_qids=len(all_qids),
    fresh_unique_normalized_question_contents=len(all_normalized),fresh_complete_trajectories=25600*len(epochs),
    actual_Adam_increment_per_rank=[100*len(epochs)]*8,actual_final_Adam_per_rank=[LAST*100]*8,
    rank_Adam_not_summed=True,epoch_facts=epochs,question_overlap_counts=pairwise,
    fixed_diagnostic_identity=fixed_qid_proof,fixed_test_manifest_sha256=next(iter(test_manifests)),
    normalized_diagnostic_exclusion_note='Frozen pool initializer excludes normalized diagnostic content. This collector directly checks diagnostic QID intersections and consistent normalized-exclusion binding; it does not independently reread all canonical question texts.',
    historical_cached_epoch1=dict(metrics=historical_metrics,metrics_proof=historical_metrics_proof,complete=historical_complete,
        complete_proof=historical_complete_proof,counted_among_fresh_epochs=False),
    current_stage=stage,current_stage_proof=stage_proof,current_latest_bundle=latest_bundle,
    latest_bundle_proof=latest_bundle_proof,expected_next_epoch_progress=next_progress,
    expected_next_epoch_actual_completed_generation_observed=next_started,
    five_fresh_plus_next_started_acceptance_complete=(len(epochs)==5 and next_started),
    small_file_readback_proofs=proofs,large_payload_or_model_independent_hash_performed=False,
    semantic_model_reload_performed=False,GPU_computation_performed=False,remote_files_written=False,
    original_question_text_or_tokens_downloaded=False,training_state_or_automation_modified=False)
print(json.dumps(result),flush=True)
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--first-epoch',type=int,required=True)
    parser.add_argument('--last-epoch',type=int,required=True)
    parser.add_argument('--expected-next-epoch',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--base',default=DEFAULT_BASE)
    parser.add_argument('--job-id',type=int,default=3408)
    args=parser.parse_args()
    if args.first_epoch<2 or args.last_epoch<args.first_epoch or args.last_epoch-args.first_epoch>9:
        parser.error('Read only a bounded range of at most10 actual fresh epochs starting from2')
    if args.expected_next_epoch!=args.last_epoch+1:parser.error('Expected next epoch must follow the last accepted epoch')
    output=args.output.resolve()
    transport=output.with_name(output.stem+'-transport.json')
    if output.exists()or transport.exists():raise FileExistsError('Existing audit output; choose a fresh tag, never overwrite')
    payload=dict(base=args.base,job_id=args.job_id,first_epoch=args.first_epoch,last_epoch=args.last_epoch,
                 expected_next_epoch=args.expected_next_epoch,runtime_source_commit=RUNTIME)
    sys.path.insert(0,REPO)
    from reproduction.h1008_qprobe.management_transport import remote
    value=json.loads(remote(REMOTE.replace('ARGS',repr(payload)),transport,control_path=None,target='h100-8'))
    value['transport_proof']=dict(path=str(transport),sha256=hashlib.sha256(transport.read_bytes()).hexdigest())
    with output.open('x')as stream:json.dump(value,stream,indent=2,sort_keys=True);stream.write('\n')
    print(json.dumps(dict(path=str(output),sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
         fresh_complete_epochs=value['fresh_complete_epochs'],actual_final_Adam_per_rank=value['actual_final_Adam_per_rank'],
         next_generation_observed=value['expected_next_epoch_actual_completed_generation_observed'],
         acceptance_complete=value['five_fresh_plus_next_started_acceptance_complete']),indent=2))


if __name__=='__main__':
    main()
