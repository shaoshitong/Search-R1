"""Remote frozen peer Prefix5/10/20 ambiguity analysis, identity exclusions first.

Metric definitions match new-extraG5-prefix-conflict-diagnostic.py. No full
response, tensors, probe probabilities, fitting or source writer is touched.
"""
import argparse
from collections import Counter,defaultdict
import gzip,hashlib,json,math,os
from pathlib import Path
import sys,time

RUN='70188b8e50fd4b45bc420ce58014f46e'
REVISION='a09a35458c702b33eeacc393d103063234e8bc28'
SOURCE='/mnt/data/shaoshitong/qwen7b-resample-seed20261021-g5-20261020'
ALIAS='/mnt/shared/workspace/training_data_node_10/shaoshitong/qwen7b-resample-seed20261021-g5-20261020'
STATUS='FROZEN_PEER_TRAIN_PREFIX_5_10_20_OWN_EM_AGGREGATE_DIAGNOSTIC_COMPLETE'


def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for data in iter(lambda:f.read(1024*1024),b''):h.update(data)
 return h.hexdigest()
def fp(p):p=Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def stat(p):
 s=Path(p).stat();return (s.st_size,s.st_ino,s.st_dev,s.st_mtime_ns,s.st_ctime_ns)
def ratio(a,b):return a/b if b else None
def entropy(positive,n):
 if not positive or positive==n:return 0.
 p=positive/n;return -p*math.log(p)-(1-p)*math.log1p(-p)
def context_at(row,n):
 segments=[]
 for segment in row['segments']:
  take=min(segment['policy_length'],max(0,n-segment['policy_start']))
  if take:segments.append([segment['turn_index'],segment['policy_start'],segment['position_start'],segment['context_tokens'][:segment['request_length']+take]])
 return json.dumps([row['position_ids'][:n],segments],sort_keys=True,separators=(',',':'))


def empty():
 return dict(Q=0,N=0,K=Counter(),mixed=0,all_cw=0,statistics={str(n):dict(hist=Counter(),duplicate_pairs=0,conflict_pairs=0,duplicated_questions=0,all_identical=0,
  conflicted_mixed=0,excess=0,has_duplicate_peer=0,short=0,ce=0.,ce_mixed=0.,context_ambiguous_clusters=0) for n in (5,10,20)})


def add(groups,total):
 for group in groups.values():
  if len(group)!=5 or {r['response_index'] for r in group}!=set(range(5)):raise ValueError('Authentic same-question own G5 required')
  if len({r['question_hash'] for r in group})!=1 or len({tuple(r['prompt']) for r in group})!=1:raise ValueError('Same-question prompt/hash differs')
  k=sum(r['label'] for r in group);is_mixed=0<k<5;total['Q']+=1;total['N']+=5;total['K'][k]+=1;total['mixed']+=is_mixed;total['all_cw']+=k*(5-k)
  for n in (5,10,20):
   s=total['statistics'][str(n)];clusters=defaultdict(list)
   for row in group:clusters[tuple(row['tokens'][:n])].append(row);s['short']+=len(row['tokens'])<n
   distinct=len(clusters);s['hist'][distinct]+=1;s['duplicated_questions']+=distinct<5;s['all_identical']+=distinct==1;s['excess']+=5-distinct
   local=0
   for own in clusters.values():
    size=len(own);positive=sum(row['label'] for row in own);s['duplicate_pairs']+=size*(size-1)//2;local+=positive*(size-positive)
    if size>1:s['has_duplicate_peer']+=size
    term=size*entropy(positive,size);s['ce']+=term
    if is_mixed:s['ce_mixed']+=term
    if size>1 and len({context_at(row,n) for row in own})>1:s['context_ambiguous_clusters']+=1
   s['conflict_pairs']+=local;s['conflicted_mixed']+=bool(local and is_mixed)


def summarize(total):
 Q,N,mixed,all_cw=(total[k] for k in ('Q','N','mixed','all_cw'));statistics={}
 for n,s in total['statistics'].items():
  statistics[n]=dict(questions=Q,trajectories=N,distinct_prefix_count_histogram={str(i):s['hist'][i] for i in range(1,6)},
   any_duplicate_prefix_questions=s['duplicated_questions'],any_duplicate_prefix_question_fraction=ratio(s['duplicated_questions'],Q),
   all_five_identical_questions=s['all_identical'],all_five_identical_question_fraction=ratio(s['all_identical'],Q),
   excess_duplicate_trajectories=s['excess'],excess_duplicate_trajectory_fraction=ratio(s['excess'],N),
   trajectories_with_duplicate_peer=s['has_duplicate_peer'],trajectory_duplicate_peer_fraction=ratio(s['has_duplicate_peer'],N),
   equal_prefix_unordered_pairs=s['duplicate_pairs'],all_within_question_unordered_pairs=10*Q,equal_prefix_pair_fraction=ratio(s['duplicate_pairs'],10*Q),
   equal_prefix_opposite_EM_pairs=s['conflict_pairs'],all_within_question_correct_wrong_pairs=all_cw,
   same_prefix_fraction_of_correct_wrong_pairs=ratio(s['conflict_pairs'],all_cw),true_Mixed_questions=mixed,
   true_Mixed_with_conflicting_prefix=s['conflicted_mixed'],true_Mixed_conflicting_prefix_question_fraction=ratio(s['conflicted_mixed'],mixed),
   token_exact_empirical_trajectory_BCE_lower_bound_nats=ratio(s['ce'],N),token_exact_empirical_Mixed_only_trajectory_BCE_lower_bound_nats=ratio(s['ce_mixed'],5*mixed),
   shorter_naturally_terminal_trajectories=s['short'],token_equal_clusters_with_different_recorded_context_or_positions=s['context_ambiguous_clusters'])
 return dict(questions=Q,trajectories=N,K_histogram=[total['K'][i] for i in range(6)],true_Mixed_questions=mixed,statistics=statistics)


def exclusion(packet):
 if packet['role']!='peer_prefix_diagnostic_eval_identity_exclusions' or packet.get('final_labels_or_rollouts_read') is not False:raise ValueError('Explicit identity-only packet required')
 qids=set();hashes=set();counts={}
 for role,count in [('diagnostic512',512),('selection2048',2048),('frozen_final4096',4096)]:
  values=packet['scopes'][role]
  if len(values)!=count:raise ValueError('Evaluation exclusion count differs')
  for row in values:
   if set(row)!={'qid','question_hash'}:raise ValueError('Never send/read evaluation labels in exclusion packet')
   qids.add(row['qid']);hashes.add(row['question_hash'])
  counts[role]=count
 return qids,hashes,counts


def run(snapshot,packet):
 root=Path(snapshot);done=read(root/'complete.json');before=fp(root/'complete.json');excluded,excluded_hashes,counts=exclusion(packet)
 if (done['status']!='SEALED_CANONICAL_NEWSEED_PREFIX20_G5_SNAPSHOT_EXPORT_COMPLETE' or done['source_run_id']!=RUN or done['revision']!=REVISION or
     done['questions']!=31232 or done['trajectories']!=156160 or done.get('fresh_all_source_files_SHA256') is not True):raise ValueError('Exact fixed accepted31232 scope required')
 total=empty();seen=set();sources=[];rejected=0;distribution=Counter();proofs=[];source_meta={}
 for archive in done['groups']:
  name=archive['group'];base=root/'payload'/name/'source';members={f['path'].split('/',2)[2]:f for f in archive['files']}
  def checked(p):
   rel='source/'+str(Path(p).relative_to(base));item=members[rel]
   if Path(p).is_symlink() or Path(p).stat().st_size!=item['bytes'] or sha(p)!=item['sha256']:raise ValueError('Frozen copied prefix/EM SHA differs')
   return item
  conf=read(base/'config.json');manifest=read(base/'inputs/manifest.json')
  source_meta=dict(source_config=fp(base/'config.json'),original_canonical_train_SHA=conf.get('input_sha256',{}).get('train'),
   source_manifest_ordered_qid_SHA=manifest.get('ordered_qid_sha256'),original169028_provenance=packet['canonical_train_proof'])
  for inp in sorted((base/'inputs').glob('shard-*/batch*.json')):
   checked(inp);batch=read(inp);allinputs={r['qid']:r for r in batch['rows']};eligible={q:r for q,r in allinputs.items() if q not in excluded and r['question_hash'] not in excluded_hashes}
   rejected+=len(allinputs)-len(eligible)
   if not eligible:continue
   dest=base/'sampling'/inp.parent.name/'batches'/inp.stem;receipt=read(dest/'complete.json');checked(dest/'complete.json')
   if receipt['run_id']!=RUN or receipt['input_sha256']!=sha(inp) or receipt['actor_optimizer_updates']!=0:raise ValueError('Frozen batch/input/actor binding differs')
   def path(item):
    raw=item['path'];relative=next((raw[len(prefix)+1:] for prefix in (SOURCE,ALIAS) if raw.startswith(prefix+'/')),None)
    if relative is None:raise ValueError('Frozen receipt escapes source namespace')
    p=base/relative;checked(p)
    if sha(p)!=item['sha256'] or p.stat().st_size!=item['bytes']:raise ValueError('Frozen producer prefix/EM SHA differs')
    return p
   prefix=path(receipt['prefix']['prefix']);em=path(receipt['full']['own_em']);initial={str(p):stat(p) for p in (prefix,em,inp)}
   labels={}
   with em.open() as f:
    for line in f:
     row=json.loads(line);qid=row['qid']
     if qid not in eligible:continue  # identity boundary precedes every EM access
     tid=row['trajectory_id']
     if tid in labels or type(row['strict_em']) is not int or row['strict_em'] not in (0,1):raise ValueError('Own EM duplicate/type differs')
     labels[tid]=row
   groups=defaultdict(list)
   with gzip.open(prefix,'rt') as f:
    for line in f:
     row=json.loads(line);qid=row['qid']
     if qid not in eligible:continue
     tid=row['trajectory_id'];label=labels.get(tid);source=eligible[qid]
     expected='%s-outer000001-c%04d-g%d'%(RUN,source['candidate_ordinal'],row['response_index'])
     if (tid!=expected or tid in seen or label is None or label['qid']!=qid or label['prefix_sha256']!=row['prefix_sha256'] or
         row['question_hash']!=source['question_hash'] or row['initial_prompt_token_ids']!=source['initial_prompt_token_ids'] or
         row['actor_version']!='official7b-'+REVISION or row['split']!='train' or row['run_id']!=RUN):raise ValueError('Own question/trajectory/prefix/EM binding differs')
     material={k:v for k,v in row.items() if k!='prefix_sha256' and not k.startswith('feature_')}
     if hashlib.sha256(canonical(material)).hexdigest()!=row['prefix_sha256']:raise ValueError('Own causal prefix digest differs')
     tokens=row['policy_token_ids']
     if len(tokens)>20 or (len(tokens)<20 and row['terminal'] is not True):raise ValueError('Natural Prefix20 length differs')
     seen.add(tid);groups[qid].append(dict(tokens=tokens,position_ids=row['policy_position_ids'],label=label['strict_em'],response_index=row['response_index'],
      question_hash=row['question_hash'],prompt=row['initial_prompt_token_ids'],segments=row['prefix_segments']))
   if len(labels)!=5*len(eligible) or set(groups)!=set(eligible):raise ValueError('Complete own eligibleG5 coverage differs')
   add(groups,total)
   for source in eligible.values():distribution[source.get('data_source','unrecorded')]+=1
   if initial!={str(p):stat(p) for p in (prefix,em,inp)}:raise ValueError('Frozen copied sources changed during read')
   proofs.append(dict(batch=ref_relative(inp,root),prefix_sha256=sha(prefix),own_em_sha256=sha(em),questions=len(eligible)))
 if total['Q']+rejected!=31232 or before!=fp(root/'complete.json'):raise ValueError('Fixed frozen cohort coverage/race differs')
 result=summarize(total);result.update(status=STATUS,source_complete=before,snapshot_id=root.name,frozen_questions=31232,excluded_before_EM=rejected,
  exclusion_scope_counts=counts,exclusion_packet_sha256=hashlib.sha256(canonical(packet)).hexdigest(),source_meta=source_meta,data_source_question_counts=dict(distribution),
  source_prefix_and_own_EM_SHA_fresh=True,all_archives_rehashed=False,source_complete_SHA_inherited=True,batches=len(proofs),source_batch_proofs_sha256=hashlib.sha256(canonical(proofs)).hexdigest(),
  strict_EM_source='authentic ownEM already source-audited; tokenizer rerun remains separate HPC3 ingest',probe_probabilities_available=False,p_j_or_q_i_invented=False,
  tensor_files_read=False,full_response_files_read=False,excluded_EM_inspected=False,final_labels_or_rollouts_read=False,source_writer_modified=False,
  sampling_scope='all fixed244 sealed batches of partial peer snapshot; not full169028 or an IID population sample',
  BCE_lower_bound_scope='token-only deterministic trajectory scorer empirical BCE minimum; not Mixed BA bound; different native contexts/positions invalidate applying it to richer inputs',
  early_terminal_handling='actual unpadded token tuple including length',time=time.time())
 return result


def ref_relative(path,root):return str(Path(path).relative_to(root))
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--snapshot',required=True);a=p.parse_args()
 if sys.platform=='darwin':p.error('Raw source analysis must stay remote')
 packet=json.loads(sys.stdin.buffer.readline(4*1024*1024));print(json.dumps(run(a.snapshot,packet)),flush=True)
if __name__=='__main__':main()
