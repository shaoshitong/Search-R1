"""Same-qualified-HF-backend capture of the existing precommitted validation G5.

This never chooses another G5, reads own EM/full responses, generates tokens or
opens untouched final pools. Only the existing new validation2048 prefix rows
are projected and causally recaptured. Durable output contains layer26 only.
"""
import argparse,fcntl,gzip,hashlib,json,os,time
from pathlib import Path
from . import replay_probe_layers as r
from .replay_bank_prepare import q1_model_binding
from .replay_bank_capture import verify_completed_chunk

B=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019')
V=B/'algorithm-improvement-20261021/validation-g10-20261021/materialized'
TARGET=B/'algorithm-improvement-20261021/validation-fixedG5-hf-replay-20261021'
SOURCE_SHA='5ea5197a841d7bd0b226a82350a2436a8ae3d860143a2ea39fe59b7ecb6837d5'
ROLE='PRECOMMITTED_NEW_VALIDATION2048_FIXED_G5_SAME_HF_REPLAY'
STATUS='NEW_VALIDATION2048_FIXED_G5_QUALIFIED_HF_REPLAY_LAYER26_COMPLETE'

def read(p):return r.read(p)
def ref(p):return r.fingerprints(p)
def digest(v):return hashlib.sha256(r.canonical(v)).hexdigest()
def checked(v):return r.checked_file(v)
def once(p,v):
 p=Path(p)
 if p.exists():
  if read(p)!=v:raise ValueError('Conflicting immutable validation feature artifact')
 else:r.atomic(p,v)
 return ref(p)

def fixed_declarations(plan,expected_questions=2048):
 """Identity-only: no group labels/scores enter this function."""
 summary=plan['summary'];groups=plan['groups']
 if (summary.get('role')!='validation' or summary.get('labels_consulted') is not False or
     summary.get('questions')!=expected_questions or summary.get('trajectories')!=5*expected_questions or
     summary.get('selected_g5_sha256')!=digest(groups)):
  raise ValueError('Existing precommitted label-blind fixedG5 identity required')
 qids=set();tids=set();result=[]
 for g in groups:
  if set(g)!={'qid','trajectory_ids'} or g['qid'] in qids:raise ValueError('Unique identity-only fixed groups required')
  own=g['trajectory_ids']
  if len(own)!=5 or len(set(own))!=5 or tids.intersection(own):raise ValueError('Every fixed question requires five distinct authentic IDs')
  qids.add(g['qid']);tids.update(own);result.extend(dict(qid=g['qid'],trajectory_id=t) for t in own)
 if len(qids)!=expected_questions:raise ValueError('Frozen validation question count differs')
 return result

def prefix_sources(audit):
 if audit.get('all_source_payload_SHA_freshly_verified') is not True or set(audit['events'])!={'event-A','event-B'}:
  raise ValueError('Accepted two-event prefix source audit required')
 result=[];seen=set()
 for event in ('event-A','event-B'):
  for batch in audit['events'][event]:
   own=batch['phase_payloads']['prefix']
   if own['path'] in seen:raise ValueError('Prefix source repeats')
   seen.add(own['path']);result.append(own)
 return result

def projected_rows(sources,wanted,synthetic_fixture=False):
 seen=set()
 for item in sources:
  if synthetic_fixture:
   actual=ref(item['path'])
   if any(actual[k]!=item[k] for k in ('path','bytes','sha256')):raise ValueError('Synthetic prefix SHA differs')
  else:checked(item)
  with gzip.open(item['path'],'rt') as stream:
   for line in stream:
    row=json.loads(line);tid=row['trajectory_id']
    if tid not in wanted:continue
    if tid in seen or row['qid']!=wanted[tid]:raise ValueError('Repeated/wrong actual selected validation prefix')
    if any(k in row for k in ('strict_em','labels','reward_model','ground_truth','response_token_ids')):
     raise ValueError('Only prefix rows without EM/full/future response are allowed')
    r.normalize_prefix(row,tid,wanted[tid],True);seen.add(tid);yield row
 if seen!=set(wanted):raise ValueError('Incomplete actual precommitted fixedG5 prefix coverage')

def batching_binding(golden,batch_size):
 if batch_size not in (16,32):raise ValueError('Explicit qualified batch16 or32 required')
 checked(golden);qualification=read(golden['path'])
 if (qualification['status']!='REAL_SCALAR_BATCHED_HF_REPLAY_AND_E4_GOLDEN_COMPARISON_COMPLETE' or
     str(batch_size) not in qualification['downstream'] or qualification['actor_updates']!=0 or
     qualification['probe_updates']!=0 or qualification['final_pool_read'] is not False):
  raise ValueError('Actual same-backend golden completion required')
 checked(qualification['config']);config=read(qualification['config']['path']);checked(config['source_publication']);pub=read(config['source_publication']['path'])
 expected=pub['files']['reproduction/operations/qwen7b_warmup/replay_batched_context.py']['sha256']
 from . import replay_batched_context as core
 if ref(core.__file__)['sha256']!=expected:raise ValueError('Actual golden batched primitive differs')
 return dict(batch_size=batch_size,padded_token_budget=16384,window_trajectories=batch_size,golden=golden,core_sha256=expected,
  acceptance='Explicit operator selected actual scalar/batched golden evidence; no native bitexact claim')

def prepare(base,code_root,commit,golden,batch_size):
 base=r.remote(base);code_root=r.remote(code_root)
 if base!=TARGET or code_root.parent!=base or len(commit)!=40:raise ValueError('New independent validation capture root and published source required')
 source=ref(V/'bank/complete.json')
 if source['sha256']!=SOURCE_SHA:raise ValueError('Require actual accepted new G10 validation, not old selection2048')
 accepted=read(source['path'])
 if (accepted['status']!='VALIDATION_ACTUAL_TWO_FRESH_G5_G10_SHA_OWN_EM_VERIFIED' or accepted['questions']!=2048 or accepted['trajectories']!=20480 or
     accepted['fitting_on_validation_allowed'] is not False or accepted['untouched_final'] is not False or
     accepted['fixed_G5_was_chosen_before_own_EM'] is not True):raise ValueError('Wrong fixed validation role or source')
 fixed=accepted['consumer_inputs']['fixed_G5_plan'];checked(fixed);plan=read(fixed['path']);declarations=fixed_declarations(plan)
 auditref=accepted['outputs']['source-audit.json'];checked(auditref);sources=prefix_sources(read(auditref['path']))
 if len(sources)!=32:raise ValueError('Actual complete new validation must supply32 sealed prefix sources')
 for item in sources:
  if not str(item['path']).startswith(str(V)+'/event-'):raise ValueError('Prefix source escaped the actual validation event')
 batching=batching_binding(golden,batch_size);q1,model_binding=q1_model_binding()
 base.mkdir(parents=True,exist_ok=True);(base/'inputs').mkdir(exist_ok=True);(base/'logs').mkdir(exist_ok=True)
 with (base/'prepare.lock').open('a') as guard:
  fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if (base/'preparation.json').exists():
   old=read(base/'preparation.json')
   if old['source_complete']!=source or old['batching']!=batching or old['source_commit']!=commit:raise ValueError('Prior immutable preparation differs')
   checked(old['selection']);checked(old['prefix_input']);return old
  path=base/'inputs/fixed-prefix.jsonl.gz';temp=path.with_suffix('.gz.partial')
  if path.exists() or temp.exists():raise ValueError('Unknown prefix projection preserved; use isolated attempt')
  wanted={x['trajectory_id']:x['qid'] for x in declarations};count=0
  with gzip.open(temp,'xb') as out:
   for row in projected_rows(sources,wanted):
    out.write(r.canonical(row)+b'\n');count+=1
    if count%512==0:r.atomic(base/'prepare-progress.json',dict(trajectories=count,expected_trajectories=10240,labels_read=False,time=time.time()))
  if count!=10240:raise ValueError('Every actual precommitted G5 trajectory must remain')
  os.replace(str(temp),str(path));prefix=ref(path)
  selection=dict(role=ROLE,questions=2048,trajectories=declarations,real_trajectories_per_question=5,label_blind_selection=True,
   prefix_tokens=20,capture_layers=[26],prefix_files=[prefix],model_revision=r.REVISION,model_files=q1['model_files'],source_model_binding=model_binding,
   source_complete=source,source_groups=accepted['consumer_inputs']['groups'],source_fixed_G5=fixed,fixed_G5_sha256=plan['summary']['selected_g5_sha256'],
   source_prefix_audit=auditref,source_commit=commit,expected_code_root=str(code_root),batching=batching,labels_read=False,
   new_G5_selected=False,actor_updates=0,new_responses_generated=0,untouched_final_pool_read=False)
  selected=once(base/'inputs/selection.json',selection)
  value=dict(status='NEW_VALIDATION_FIXED_G5_PREFIX_ONLY_PREPARED_NOT_CAPTURED',source_complete=source,source_groups=selection['source_groups'],
   source_fixed_G5=fixed,source_prefix_audit=auditref,selection=selected,prefix_input=prefix,source_prefix_files=sources,
   source_commit=commit,code_root=str(code_root),batching=batching,questions=2048,trajectories=10240,capture_layers=[26],
   labels_read=False,new_G5_selected=False,actor_updates=0,new_responses_generated=0,final_pool_read=False,feature_bank_created=False,time=time.time())
  once(base/'preparation.json',value);return value

def load_selection(selection,synthetic_fixture=False,expected_questions=2048):
 declarations=selection['trajectories'];wanted={x['trajectory_id']:x['qid'] for x in declarations}
 if (selection['role']!=ROLE or selection['questions']!=expected_questions or len(wanted)!=5*expected_questions or len(wanted)!=len(declarations) or
     selection['capture_layers']!=[26] or selection['label_blind_selection'] is not True or selection['new_G5_selected'] is not False):
  raise ValueError('Actual unchanged new validation fixedG5 selection required')
 found={x['trajectory_id']:r.normalize_prefix(x,x['trajectory_id'],wanted[x['trajectory_id']],True)
  for x in projected_rows(selection['prefix_files'],wanted,synthetic_fixture)}
 result=[found[x['trajectory_id']] for x in declarations];counts={}
 for row in result:counts[row['qid']]=counts.get(row['qid'],0)+1
 if len(counts)!=expected_questions or set(counts.values())!={5}:raise ValueError('Exactly original fixedG5 per question required')
 return result

def capture(selection_path,selection_sha,output):
 import torch
 from transformers import AutoModel
 from .replay_batched_context import BatchedContextCapture
 if not torch.cuda.is_available() or not os.environ.get('SLURM_JOB_ID') or not os.environ.get('CUDA_VISIBLE_DEVICES'):raise ValueError('Actual owned CUDA Slurm capture required')
 source=ref(r.remote(selection_path));selection=read(source['path']);base=r.remote(output)
 if source['sha256']!=selection_sha or selection['expected_code_root']!=str(Path(__file__).resolve().parents[3]):raise ValueError('Immutable selection/code root differs')
 if ref(r.__file__)['sha256']!=selection['source_model_binding']['legacy_replay_module_sha256']:raise ValueError('Q1 causal normalization primitive differs')
 batching=batching_binding(selection['batching']['golden'],selection['batching']['batch_size'])
 if batching!=selection['batching']:raise ValueError('Qualified batching binding differs')
 base.mkdir(parents=True,exist_ok=True)
 with (base/'capture.lock').open('a') as guard:
  fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if (base/'complete.json').exists():raise ValueError('Completed fixed validation feature bank preserved')
  selected=load_selection(selection);model_dir=selection['source_model_binding']['official_view'];proofs,config=r.verify_model(model_dir,selection)
  if any(t>=config['vocab_size'] for row in selected for seg in row['segments'] for t in seg['context_tokens']):raise ValueError('Actual token outside official vocabulary')
  device=torch.device('cuda:0');torch.cuda.set_device(device)
  model=AutoModel.from_pretrained(model_dir,torch_dtype=torch.bfloat16,attn_implementation='flash_attention_2',local_files_only=True,trust_remote_code=False,low_cpu_mem_usage=True).eval().requires_grad_(False).to(device)
  engine=BatchedContextCapture(model,batch_size=batching['batch_size'],max_context_tokens=batching['padded_token_budget'],device=device)
  started=time.monotonic();chunks=[]
  for start in range(0,len(selected),320):
   own=selected[start:start+320];root=base/('chunk-%05d'%(start//320));tids=[row['trajectory_id'] for row in own]
   if (root/'complete.json').exists():
    done=verify_completed_chunk(root,tids)
    if done['source_selection']!=source or done['batching']!=batching:raise ValueError('Existing feature chunk source/batch differs')
    chunks.append(ref(root/'complete.json'));continue
   if root.exists():raise ValueError('Unknown partial feature chunk preserved; isolated recovery required')
   root.mkdir();writer=r.ReplayLayerWriter(root/'layer-26',26,own)
   for index,(row,values) in enumerate(engine.iter_replay(own,window_trajectories=batching['window_trajectories'])):
    writer.write(row,values[26]);r.atomic(base/'progress.json',dict(status='ACTUAL_NEW_VALIDATION_FIXED_G5_HF_REPLAY_ADVANCING',trajectories=start+index+1,expected_trajectories=10240,
     completed_chunks=len(chunks),elapsed_seconds=time.monotonic()-started,batching=engine.stats(),actor_updates=0,time=time.time()))
   manifest=writer.finish();once(root/'complete.json',dict(source_selection=source,trajectory_ids=tids,feature_manifests={'26':manifest},source_runtime='huggingface_frozen_causal_prefill',batching=batching,actor_updates=0));chunks.append(ref(root/'complete.json'))
  manifests=[];ids=[]
  for item in chunks:
   checked(item);done=verify_completed_chunk(Path(item['path']).parent,read(item['path'])['trajectory_ids']);ids+=done['trajectory_ids'];manifests.append(done['feature_manifests']['26'])
  if ids!=[row['trajectory_id'] for row in selected]:raise ValueError('Whole fixedG5 actual feature identity coverage incomplete')
  from .replay_feature_reader import ReplayFeatureReader
  reader=ReplayFeatureReader(manifests,26)
  if set(reader.records)!=set(ids):raise ValueError('Final HF reader coverage differs')
  reader.get([ids[0],ids[-1]],'cpu');reader.close();features=once(base/'features-layer26.json',manifests)
  value=dict(status=STATUS,source_selection=source,source_complete=selection['source_complete'],source_groups=selection['source_groups'],source_fixed_G5=selection['source_fixed_G5'],
   fixed_G5_sha256=selection['fixed_G5_sha256'],feature_manifests_by_capture_layer={'26':manifests},features=features,chunks=chunks,questions=2048,trajectories=10240,
   capture_layers=[26],human_block_layers=[27],source_runtime='huggingface_frozen_causal_prefill',attention_implementation='flash_attention_2',model_revision=r.REVISION,model_files=proofs,
   batching=batching,actual_batching=engine.stats(),source_code=ref(__file__),actor_frozen=True,actor_updates=0,probe_updates=0,new_responses_generated=0,
   native_bitexact_equivalence_claim=False,labels_read=False,new_G5_selected=False,final_pool_read=False,job_id=os.environ['SLURM_JOB_ID'],elapsed_seconds=time.monotonic()-started,time=time.time())
  once(base/'complete.json',value);return value

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--action',choices=('prepare','capture'),required=True);p.add_argument('--base',default=str(TARGET));p.add_argument('--code-root');p.add_argument('--commit');p.add_argument('--golden');p.add_argument('--batch-size',type=int);p.add_argument('--selection');p.add_argument('--selection-sha256');p.add_argument('--output');a=p.parse_args()
 try:
  if a.action=='prepare':v=prepare(a.base,a.code_root,a.commit,ref(a.golden),a.batch_size)
  else:v=capture(a.selection,a.selection_sha256,a.output)
 except BaseException as e:
  if a.action=='capture' and a.output:
   b=Path(a.output);b.mkdir(parents=True,exist_ok=True);r.atomic(b/'failure.json',dict(error_type=type(e).__name__,error=str(e)[:1000],time=time.time()))
  raise
 print(json.dumps(v,sort_keys=True),flush=True)
if __name__=='__main__':main()
