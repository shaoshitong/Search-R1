"""Resolve canonical own-G10 by physical seal before reading retained RAW.

Protocol trajectory IDs alone are insufficient: donor and HPC3 executions can
share them and even their first20 tokens. Chosen whole-G5 execution takes
precedence; otherwise require one unique sealed payload, never select by EM.
"""
from array import array
import gc,hashlib,json,time
from pathlib import Path
from .prepare_full169028_policy80 import A,FULL,canonical,checked,proof,read,atomic

def events(x):
 if isinstance(x,dict):
  if isinstance(x.get('full'),dict)and {'path','bytes','sha256'}<=x['full'].keys()and isinstance(x.get('own_em'),dict):yield x
  for v in x.values():yield from events(v)
 elif isinstance(x,list):
  for v in x:yield from events(v)
def core(ref):return {k:ref[k]for k in('path','bytes','sha256')}
def payload(ref):return(ref['bytes'],ref['sha256'])
def seal_event(ref,expected_execution=None):
 seal=read(checked(ref));f=seal['full'];full=core(f['full']);own=core(f['own_em']);execution=seal.get('execution_run_id')or f.get('execution_run_id')or f['run_id']
 if expected_execution is not None:assert execution==expected_execution,'Chosen physical execution differs'
 return dict(full=full,own_em=own,seal=core(ref),execution_run_id=execution,chosen=True)
def resolve(base,groups,qids,lookup,labels,progress=None):
 base=Path(base);descriptor=read(FULL/'full-training-descriptor.json');canonical_complete=A/'joint169028-native-training-bank-v1/complete.json';complete=read(canonical_complete);assert complete['groups']==descriptor['groups'];events_by_path={};metadata=[]
 for ref in descriptor['source_metadata']:
  x=read(checked(ref));metadata.append(ref)
  for e in events(x):
   e=dict(full=core(e['full']),own_em=core(e['own_em']),seal=core(e['seal']),event=e.get('event'),source_metadata=ref)
   old=events_by_path.setdefault(e['full']['path'],e);assert old['full']==e['full']and old['own_em']==e['own_em']
 assert len(events_by_path)==2678
 chosen={};special_questions=set();origin_banks=[]
 # Top-level newly added banks explicitly bind whole old/newG5 physical seals.
 # The first124194 bank remains its qualified canonical source; any duplicate
 # physical payload for those IDs is rejected unless full size/SHA are equal.
 for ref in complete['source_refs']:
  if ref['key']=='current124194':continue
  gs=read(checked(ref['groups']));gs=gs['train']if 'train'in gs else gs;gs=list(gs.values())if isinstance(gs,dict)else gs;origin_banks.append(dict(key=ref['key'],groups=ref['groups'],complete=ref['complete'],questions=len(gs)))
  for g in gs:
   q=g['qid'];assert q in groups and all(groups[q][k]==g[k]for k in('qid','trajectory_ids','labels','question_hash'))
   if 'chosen_new5_seal'not in g:continue
   special_questions.add(q)
   for half,sref,execution in((0,g['old5_seal'],None),(1,g['chosen_new5_seal'],g['chosen_new5_execution_run_id'])):
    key=(sref['path'],sref['sha256'])
    if key not in chosen:chosen[key]=seal_event(sref,execution)
    e=chosen[key];path=e['full']['path'];old=events_by_path.setdefault(path,e);assert old['full']==e['full']and old['own_em']==e['own_em'];events_by_path[path]=dict(old,seal=e['seal'],execution_run_id=e['execution_run_id'],chosen=True)
    for t in g['trajectory_ids'][half*5:half*5+5]:
     i=lookup[t];assert labels[i]==g['labels'][half*5+g['trajectory_ids'][half*5:half*5+5].index(t)];chosen[(i,)]=path
  del gs;gc.collect()
 # Do not hold seals once their immutable selected payload paths are fixed.
 sources=sorted(events_by_path.values(),key=lambda e:e['full']['path']);source_index={e['full']['path']:i for i,e in enumerate(sources)};selected=array('I',[0])*len(lookup);explicit=array('I',[0])*len(lookup)
 for key,path in chosen.items():
  if len(key)==1:explicit[key[0]]=source_index[path]+1
 del chosen;gc.collect();seen=bytearray(len(lookup));candidate_labels_different=0;wrong_physical_rows=0;duplicate_equal_payload=0;unique=0;source_stats=[];used=[0]*len(sources);started=time.time()
 for si,e in enumerate(sources):
  fresh=checked(e['own_em']);count=0;skipped=0
  with fresh.open()as f:
   for line in f:
    row=json.loads(line);t=row['trajectory_id']
    if t not in lookup:continue
    i=lookup[t];q=qids[i//10];assert row['qid']==q and row['actor_version']=='official7b-a09a35458c702b33eeacc393d103063234e8bc28'and type(row['strict_em'])is int and row['strict_em']in(0,1)
    if explicit[i]and explicit[i]!=si+1:
     wrong_physical_rows+=1;skipped+=1;candidate_labels_different+=row['strict_em']!=labels[i];continue
    if selected[i]:
     prev=sources[selected[i]-1];assert payload(prev['full'])==payload(e['full']),'Ambiguous different physical payload for unqualified canonical own tid: '+t;assert row['strict_em']==labels[i];duplicate_equal_payload+=1;continue
    assert row['strict_em']==labels[i],'Chosen physical own_EM differs from canonical: '+t
    selected[i]=si+1;seen[i]=1;unique+=1;count+=1;used[si]+=1
  source_stats.append(dict(source_index=si,selected_rows=count,excluded_wrong_physical_rows=skipped,own_em=e['own_em'],full=e['full']))
  if progress and(si+1)%16==0:progress('RESOLVING_CHOSEN_PHYSICAL_SEAL_AND_OWN_EM',si+1,physical_sidecar_sources_total=len(sources),physical_selected_rows=unique,excluded_same_protocol_tid_wrong_execution=wrong_physical_rows,seconds=time.time()-started)
 assert unique==1690280 and all(seen)
 # Every original G5 is from one physical full payload; do not mix executions.
 for qi,q in enumerate(qids):
  for half in(0,1):
   indexes=selected[10*qi+5*half:10*qi+5*half+5];assert len(set(payload(sources[i-1]['full'])for i in indexes))==1,'Own half-G5 was mixed across physical payloads'
 active=[];reindex={};seal_proofs=[]
 for i,e in enumerate(sources):
  if not used[i]:continue
  seal=read(checked(e['seal']));f=seal['full'];assert payload(f['full'])==payload(e['full'])and payload(f['own_em'])==payload(e['own_em']),'Source physical seal payload binding differs';execution=seal.get('execution_run_id')or f.get('execution_run_id')or f['run_id'];e=dict(e,execution_run_id=execution,selected_rows=used[i]);reindex[i+1]=len(active);active.append(e);seal_proofs.append(e['seal'])
 for i,x in enumerate(selected):selected[i]=reindex[x]
 path=base/'physical-source-index-by-sorted-canonical-own-tid.bin';path.write_bytes(selected.tobytes());contract=dict(status='ALL169028_OWN_G10_PHYSICAL_SEAL_EXECUTION_AND_OWN_EM_RESOLVED',canonical_complete=proof(canonical_complete),canonical_groups=descriptor['groups'],source_metadata=metadata,source_banks=origin_banks,selected_sources=active,selected_source_index=proof(path),source_index_dtype='native_uint32',source_index_order='sorted canonical qid; original own-G10 trajectory_ids',training_questions=169028,training_trajectories=1690280,explicit_chosen_physical_questions=len(special_questions),excluded_same_protocol_tid_wrong_physical_rows=wrong_physical_rows,excluded_wrong_execution_label_disagreements=candidate_labels_different,duplicate_equal_payload_rows=duplicate_equal_payload,all_selected_own_EM_matches_canonical=True,actual_physical_execution_and_seal_source_exact=True,whole_G5_not_mixed=True,selection_used_EM=False,ambiguous_different_payloads_allowed=False,canonical_labels_changed=False,final4096_read=False)
 atomic(base/'physical-source-contract.json',contract);atomic(base/'physical-source-sidecar-audit.json',dict(sources=source_stats,all_canonical_selected_labels_exact=True,final4096_read=False));return active,selected,proof(base/'physical-source-contract.json')
