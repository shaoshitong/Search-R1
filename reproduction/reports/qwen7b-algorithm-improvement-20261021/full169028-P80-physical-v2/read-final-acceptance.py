"""Final small metadata/SHA acceptance, no duplicate RAW/token scan or GPU."""
import fcntl,hashlib,json,time
from pathlib import Path
B=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/full169028-policy80-physical-ownG10-20261023-v2')
def read(p):return json.loads(Path(p).read_bytes())
def proof(p):
 p=Path(p);h=hashlib.sha256()
 with p.open('rb')as f:
  for z in iter(lambda:f.read(8388608),b''):h.update(z)
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def check(ref):assert proof(ref['path'])=={k:ref[k]for k in('path','bytes','sha256')};return read(ref['path'])
f=B/'fixture';a=read(f/'accepted.json');c=read(B/'complete.json');d=check(a['fixture']);assert c['acceptance']==proof(f/'accepted.json')and c['fixture']==a['fixture'];assert a['status']=='FULL169028_AUTHENTIC_POLICY80_OWN_G10_FIXED2048_ACCEPTED'and d['status']=='FULL169028_AUTHENTIC_POLICY80_OWN_G10_FIXED2048_READY';assert d['training_questions']==169028 and d['training_trajectories']==1690280 and d['validation_questions']==2048 and d['validation_trajectories']==10240 and d['max_real_policy_prefix']==80
assert a['actual_physical_execution_and_seal_source_exact']and a['all_train_labels_exact_canonical']and a['all_RAW_derived_first80_and_prompt_readback_equal'];p=check(a['physical_source_contract']);assert p['status']=='ALL169028_OWN_G10_PHYSICAL_SEAL_EXECUTION_AND_OWN_EM_RESOLVED'and p['actual_physical_execution_and_seal_source_exact']and p['all_selected_own_EM_matches_canonical']and p['whole_G5_not_mixed']and not p['selection_used_EM']and not p['canonical_labels_changed'];assert proof(p['selected_source_index']['path'])==p['selected_source_index']
assert a['fresh_full_P20_prefix_prompt_and_own_EM_exact'];q=check(a['P20_alignment']);assert q['status']=='FULL169028_AUTHENTIC_POLICY80_ALL_P20_PREFIX_PROMPT_OWN_EM_CROSSCHECK_ACCEPTED'and q['all_train_own_trajectory_ids']==1690280 and q['all_initial_prompt_token_ids_exact']and q['all_true_retained_policy_prefix20_exact']and q['all_label_associations_exact'];check(q['producer_acceptance']);check(a['source_contract']);check(q['source_publication'])
assert d['validation_groups']['sha256']=='50f1e3a6927d02c0f8cadf1feb5b4229b075b5d099fe31fd0de5c2ae3b149292'and d['fixed_train_G5']['sha256']=='e6e74d92160efb0740ba5c0316c4ee7686f06a499c90cafd5a545f325b62209b';check(d['validation_groups']);check(d['fixed_train_G5']);assert d['rows_by_rank']==[850260,850260]and len(d['token_shards'])%2==0
processes=[]
for name,needle in(('prepare-process.json',b'prepare_full169028_policy80_physical_v2.py'),('P20-crosscheck-process.json',b'crosscheck_full169028_policy80_P20.py')):
 r=read(B/name);proc=Path('/proc/%d'%r['pid']);live=False
 if proc.exists():live=(proc/'stat').read_text().rsplit(')',1)[1].split()[19]==r['start_ticks']and needle in(proc/'cmdline').read_bytes()
 processes.append(dict(pid=r['pid'],actual_identity_live=live,process_receipt=proof(B/name)))
locks=[]
for name in('prepare.lock','P20-crosscheck.lock'):
 h=(B/name).open('r');free=True
 try:fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:free=False
 h.close();locks.append(dict(path=str(B/name),free=free))
assert all(not p['actual_identity_live']for p in processes)and all(p['free']for p in locks),'Final receipt exists but writer exit/lock release still pending'
print(json.dumps(dict(status='FULL169028_PHYSICAL_POLICY80_PLUS_COMPLETE_P20_ALIGNMENT_FINALLY_ACCEPTED',time=time.time(),fixture=proof(f/'fixture-complete.json'),acceptance=proof(f/'accepted.json'),complete=proof(B/'complete.json'),physical_source_contract=a['physical_source_contract'],P20_alignment=a['P20_alignment'],training_questions=169028,training_trajectories=1690280,validation_questions=2048,validation_trajectories=10240,token_shards=len(d['token_shards']),rows_by_rank=d['rows_by_rank'],selected_physical_RAW_sources=len(p['selected_sources']),excluded_wrong_execution_trajectories=p['excluded_same_protocol_tid_wrong_physical_rows'],canonical_labels_changed=False,pilot_train_P80_differing_rows=d['pilot_train_own_policy80_differing_rows'],writers=processes,locks=locks,all_payload_SHA_readback_inherited_from_actual_finished_producer_and_independent_audit=True,duplicate_RAW_or_token_scan_here=False,Slurm_submitted=False,no_GPU=True,final4096_read=False)))
