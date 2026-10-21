"""One owned H100 sequentially captures only source-resolved fixed14592 G10.

Standalone entry imports the existing immutable243 qualified capture closure.
No actor/probe updates, response generation, postblock cache or donor merge.
"""
import argparse,fcntl,gzip,hashlib,json,os,sys,time
from pathlib import Path
from prepare_hpc3_tail14592_l26 import TARGET,QUALIFIED,PUB,RAW,GOLDEN,fp,checked,read,once,canonical
STATUS='FIXED_HPC3_TAIL14592_REAL_G10_QUALIFIED_HF_LAYER26_SHARD_COMPLETE'
def qualified():
 pub=read(checked(PUB))
 for name,item in pub['files'].items():
  ref=fp(QUALIFIED/name)
  if ref['bytes']!=item['bytes']or ref['sha256']!=item['sha256']:raise ValueError('Qualified runtime drift '+name)
 sys.path.insert(0,str(QUALIFIED))
 from reproduction.operations.qwen7b_warmup import peer_audited122_l26_replay as core
 from reproduction.operations.qwen7b_warmup import replay_probe_layers as r
 return core,r

def capture(preparation_sha):
 import torch
 if not os.environ.get('SLURM_JOB_ID')or not torch.cuda.is_available()or torch.cuda.device_count()!=1:raise ValueError('Owned oneH100 allocation required')
 core,r=qualified();pr=fp(TARGET/'preparation.json');assert pr['sha256']==preparation_sha;prep=read(pr['path']);assert prep['questions']==14592 and prep['trajectories']==145920 and prep['qualified_runtime_publication']==PUB
 assert fp(RAW)==prep['source_raw_complete']and all(fp(x['path'])==x for x in prep['selections'])
 with(TARGET/'single-gpu-capture.lock').open('a+')as guard:
  fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);results=[]
  for i,ref in enumerate(prep['selections']):
   sel=read(ref['path']);assert sel['shard']==i and sel['questions']==3648 and sel['source_ingest']==prep['source_raw_complete']and sel['expected_code_root']==str(QUALIFIED)
   output=TARGET/('shard-%02d'%i)/'capture';core.capture(ref['path'],ref['sha256'],output,expected_questions=3648,complete_status=STATUS);results.append(fp(output/'complete.json'))
  return dict(status='FIXED14592_FOUR_REAL_LAYER26_SHARDS_CAPTURED_PENDING_SOURCE_BOUND_BANK_JOIN',preparation=pr,shards=results,actor_updates=0,probe_updates=0,job_id=os.environ['SLURM_JOB_ID'])

def merge():
 # Metadata-only. Full binary SHA/reader checks were executed by owned producers.
 core,r=qualified();from_reader=True
 from reproduction.operations.qwen7b_warmup.replay_feature_reader import ReplayFeatureReader
 pr=fp(TARGET/'preparation.json');prep=read(pr['path']);assert prep['questions']==14592 and prep['trajectories']==145920 and fp(RAW)==prep['source_raw_complete'];truth=read(checked(read(RAW)['outputs']['groups']))['train'];lookup={t:g['qid']for g in truth for t in g['trajectory_ids']};assert len(lookup)==145920
 out=TARGET/'bank';out.mkdir(exist_ok=True)
 with(out/'merge.lock').open('a+')as guard:
  fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if(out/'complete.json').exists():
   done=read(out/'complete.json');assert done['preparation']==pr
   for x in done['outputs'].values():checked(x)
   return done
  refs=[];allids=set();bindings=[];packed=0;source_receipts=[]
  for i,selection in enumerate(prep['selections']):
   sel=read(checked(selection));cap=TARGET/('shard-%02d'%i)/'capture/complete.json';done=read(cap);assert done['status']==STATUS and done['source_selection']==selection and done['questions']==3648 and done['trajectories']==36480 and done['persist_layers']==[26]and done['batching']==prep['batching']and done['actor_updates']==done['probe_updates']==done['new_responses_generated']==0
   physical=read_bindings(checked(sel['chosen_physical_prefix_bindings']));expected={x['trajectory_id']:x['qid']for x in sel['trajectories']};seen=set();side=[]
   for ref in done['feature_manifests_by_capture_layer']['26']:
    m=read(checked(ref));assert m['capture_layer']==26 and m['prefix_tokens']==20 and m['question_max_tokens']==273 and m['dimension']==3584 and m['model_revision']=='a09a35458c702b33eeacc393d103063234e8bc28'and m['layout']=='lossless_question_base_xor_policy_v1'
    for f in m['files'].values():assert Path(f['path']).stat().st_size==f['bytes'];packed+=f['bytes']
    reader=ReplayFeatureReader([ref],26)
    index=checked(m['files']['index.jsonl']);n=0
    for line in index.open():
     row=json.loads(line);t=row['trajectory_id'];assert t not in seen and t not in allids and expected.get(t)==row['qid']==lookup.get(t);binding=physical[t];assert row['source_prefix_sha256']==binding['prefix_body_sha256'];assert len(row['retained_feature_sha256'])==64
     side.append(dict(binding,feature_manifest=ref,feature_index=m['files']['index.jsonl'],feature_retained_row_sha256=row['retained_feature_sha256']));seen.add(t);n+=1
    assert n==m['count'];ids=list(reader.records)
    for start in range(0,len(ids),8):
     actual=reader.get(ids[start:start+8],'cpu');assert actual['trajectory_ids']==ids[start:start+8];del actual
    reader.close();refs.append(ref)
    r.atomic(TARGET/'bank-acceptance-progress.json',dict(stage='FRESH_PAYLOAD_SHA_ALL_RETAINED_ROWS_AND_PHYSICAL_SOURCE_BINDING_ADVANCING',manifests=len(refs),retained_rows=sum(read(x['path'])['count']for x in refs),expected=145920,actor_updates=0,time=time.time()))
   assert seen==set(expected)==set(physical);allids.update(seen);path=out/('source-bound-retained-rows-shard-%02d.jsonl.gz'%i)
   if path.exists():raise ValueError('Unfinished prior metadata join preserved; do not overwrite')
   with gzip.open(path,'xb')as f:
    for item in side:f.write(canonical(item)+b'\n')
   bindings.append(fp(path));source_receipts.append(fp(cap))
  assert allids==set(lookup);outputs={};outputs['groups.json']=once(out/'groups.json',dict(train=truth));outputs['features-layer26.json']=once(out/'features-layer26.json',refs);outputs['source-bound-retained-rows.json']=once(out/'source-bound-retained-rows.json',bindings)
  result=dict(status='FIXED_HPC3_TAIL14592_OWN_G10_LAYER26_SOURCE_RESOLVED_FEATURE_BANK_ACCEPTED',questions=14592,trajectories=145920,preparation=pr,raw_complete=prep['source_raw_complete'],source_resolution_contract=prep['source_resolution_contract'],shard_completes=source_receipts,qualified_runtime_publication=PUB,outputs=outputs,capture_layer=26,prefix_tokens=20,question_max_tokens=273,model_revision='a09a35458c702b33eeacc393d103063234e8bc28',batching=prep['batching'],manifests=len(refs),packed_bytes=packed,physical_execution_prefix_and_retained_row_SHA_binding_complete=True,donor_feature_union_performed=False,producer_full_payload_SHA_preserved=True,metadata_join_rehashes_tensor_payload=True,all_retained_rows_reconstructed_SHA_and_finite_verified=True,actor_updates=0,probe_updates=0,new_responses_generated=0,current124194_fit_expanded=False,time=time.time());once(out/'complete.json',result);return result

def read_bindings(path):
 with gzip.open(path,'rt')as f:rows=[json.loads(line)for line in f]
 d={r['trajectory_id']:r for r in rows}
 if len(d)!=len(rows):raise ValueError('Repeated physical prefix binding ID')
 return d
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--action',choices=('capture','merge'),required=True);p.add_argument('--preparation-sha256');a=p.parse_args();print(json.dumps(capture(a.preparation_sha256)if a.action=='capture'else merge(),sort_keys=True))
