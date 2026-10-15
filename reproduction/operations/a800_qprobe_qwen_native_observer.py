"""One bounded metadata-only primary read of an authorized native-Qwen run.

Reuse published fresh-zero four-rank counting/window readers byte-for-byte.
No tensor/QA/token read, model import, evaluation, verifier, GPU query or retry.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

sys.dont_write_bytecode=True
ROOT=Path('/Users/shaoshitong/Documents/Search-G1')
OPS=Path(__file__).resolve().parents[2]
READERS=(
 ('reproduction/operations/a800_qprobe_warmup10_fresh_observer.py','331e35825d696e28dbd486759f1c2a21e4cce50a','f06ae4efeac461423eecd21e374584b8ceb94f7a9b1f9dd5dd169fe9c26807e2'),
 ('reproduction/operations/a800_qprobe_milestone_observer.py','ef1945b6b3ac1ed6cc5f2463edd9e639e7f8444c','bd58a2e67f7740643fc19287a58ff50f0cb3b0b81c9d8c04bbbbd43aa5877182'),
 ('reproduction/operations/a800_qprobe_flex_observer.py','d78c9d3d111adef8ee048b7d3cda4d50dc6066e2','e1dda0a55243ac0d25ba223b3cd95acdf37725ab09da051558ce93cfb70317cf'))


REMOTE_NATIVE=r'''
def native_version(value,outer):
 need(type(value) is str and re.fullmatch('qprobe-qwen:'+str(outer)+':'+str(outer)+':[0-9a-f]{16}',value),'native_head_version')
 return value
def native_adam(outer):
 window=min(outer,8)
 return 200+4*(window*(window+1)//2+8*max(0,outer-8))
def validate_native_recipe(q):
 expected=dict(head_kind='qwen_native_lastblock_binary_v1',loss_mode='token_binary_std_v1',ranking='binary_mixed_probability',
  candidate_questions=5120,selected_questions=512,responses_per_question=5,exploration_questions=0,
  feature_dim=2048,hidden_size=2048,head_hidden_size=2048,num_attention_heads=16,num_key_value_heads=2,
  intermediate_size=11008,source_layer=35,num_hidden_layers=36,rope_theta=1000000.0,rms_norm_eps=1e-6,
  max_position_embeddings=32768,attention_implementation='sdpa',max_tokens=244,micro_batch_size=32,head_device='cuda:0',
  head_epochs_per_outer=2,bootstrap_head_epochs=10,head_batch_size=256,head_learning_rate=.0003,
  std_loss_weight=.3,mean_loss_weight=0.,count_aux_loss_weight=0.,target_statistic='sample_std',
  seed=20261013,head_seed=1729,replay_window_outer=8,teacher_replay_excluded=True)
 need(type(q) is dict and all(q.get(k)==v and not(type(q.get(k)) is bool and type(v) is not bool) for k,v in expected.items()),'native_recipe')
 need(q.get('replay_half_life_outer') is None,'native_replay_has_no_half_life_renormalization')
 return q
def pinned_run():
 global base,repo,run,source_map_sha,run_manifest_sha
 intent=payload['intent'];base=Path(intent['base']);repo=base/'repo'
 need(intent['role']=='a800_qwen_native_formal_prospective_intent' and intent['host']=='baai-a800','native_formal_intent')
 need(re.fullmatch('[0-9a-f]{32}',intent['deployment_id']) and str(base)=='/data/shaoshitong/Search-G1-a800-fresh200-'+intent['deployment_id'],'native_base')
 need(re.fullmatch('[0-9a-f]{40}',intent['source_commit']) and intent['target_updates']==200 and
  intent['native_formal_200_authorized'] is True and intent['source_publication_authorized'] is True,'native_source_authority')
 value,info=document('preparation/intent.json');need(value==intent,'remote_intent_binding')
 run,info=document('logs/run-manifest.json');need(run is not None,'run_manifest_missing');run_manifest_sha=info['scanned_sha256']
 need(run['run_id']==payload['expected_run'] and run['execution_id']==payload['expected_execution'] and
  run['source_commit']==intent['source_commit'] and run['source_sha256']==intent['source_sha256'],'actual_execution_binding')
 source_map_sha=digest(canonical(run['source_sha256']))
 need(run.get('full_state_restore') is None and run['completed_updates']==0 and run['target_updates']==200 and
  run['training_world_size']==4 and run['actor_micro_batch_per_rank']==4 and run['budget']['ppo_micro_batch_size']==16 and
  run['budget']['per_rank_actor_micro_batch']==4 and run['budget']['ppo_mini_batch_size']==128,'fresh_four_rank_micro4')
 need(run['initial_model']==run['reference_model']=='/data/search-g1/assets/qwen2.5-3b-instruct' and
  run['execution_backend']=='ssh_local' and run.get('slurm_job_id') is None and run['execution_host']==socket.gethostname(),'fresh_official_local_actor')
 validate_native_recipe(run['question_probe'])
 need(subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True,timeout=10).strip()==intent['source_commit'],'runtime_git_head')
 need(not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],timeout=10).strip(),'runtime_git_clean')
 actual={}
 for path in repo.rglob('*'):
  if not path.is_file():continue
  relative=path.relative_to(repo);name=str(relative)
  if {'.git','__pycache__','evidence'}.intersection(relative.parts):continue
  if path.suffix not in ('.py','.sh','.yaml','.yml','.json') and name not in ('verl/version/version','verl/single_controller/version/version'):continue
  if name not in intent['source_sha256'] and re.fullmatch(r'outputs/\d{4}-\d{2}-\d{2}/\d{2}-\d{2}-\d{2}/\.hydra/(config|hydra|overrides)\.yaml',name):continue
  actual[name]=digest(budget_bytes(path,4*1024*1024))
 need(actual==intent['source_sha256'],'actual_runtime_source_bytes')
 publication,publication_info=document('preparation/source-publication.json');need(publication is not None and
  publication_info['scanned_sha256']==intent['source_publication_sha256'] and publication['source_commit']==publication['remote_commit']==intent['source_commit'] and
  publication['source_map_sha256']==source_map_sha and publication['remote_fresh_object_store_disk_git_hash_equal'] is True,'published_runtime_binding')
 release,release_info=document('preparation/native-formal-release-review.json');qualified,qualified_info=document('logs/native-prelaunch.json')
 claim,claim_info=document('logs/native-formal-launch-once.json');launch,launch_info=document('logs/deployment-launch-receipt.json')
 need(all(x is not None for x in (release,qualified,claim,launch)),'approved_actual_launch_metadata_missing')
 need(release['role']=='native_qwen_formal200_independent_release_review' and
  all(release.get(k) is True for k in ('approved','independent','context_free','authorized_source_publication_approved','actual_deployment_review_approved')) and
  release['reviewer']!=release['implementation_actor'] and release['source_commit']==intent['source_commit'] and
  release['source_map_sha256']==source_map_sha and release['human_intent_sha256']==intent['human_intent_sha256'],'context_free_release_gate')
 need(qualified['stage']=='NATIVE_FRESH_FORMAL200_PRELAUNCH_QUALIFIED' and qualified['run_id']==run['run_id'] and
  qualified['execution_id']==run['execution_id'] and qualified['source_commit']==intent['source_commit'] and qualified['source_map_sha256']==source_map_sha,'prelaunch_identity')
 for key in ('warmup_ready_sha256','warmup_head_sha256','bank_manifest_sha256'):need(release[key]==qualified[key],'release_asset_binding')
 same_execution(claim);same_execution(launch)
 need(claim['review_sha256']==release_info['scanned_sha256'] and launch['deployment_id']==intent['deployment_id'],'actual_launch_review_binding')
 return run
def native_initialization():
 q=run['question_probe'];path=Path(q['initial_head_ready']);relative=str(path.relative_to(base))
 need(relative=='artifacts/question-probe/native-qwen-warmup-adoption/ready.json','native_initial_ready_namespace')
 value,info=document(relative);need(value is not None and info['scanned_sha256']==q['initial_head_ready_sha256'],'native_initial_ready_sha')
 receipt=value['head_receipt'];provenance=value['initialization_provenance'];warmup=value['actual_warmup'];proof=value['native_block_proof']
 need(value['stage']=='QWEN_NATIVE_FORMAL_INITIAL_HEAD_ADOPTED' and value['run_id']==run['run_id'] and value['source_commit']==run['source_commit'] and
  value['completed_outer']==0 and value['actual_head_adam_updates']==200 and value['warmup_questions']==5120 and value['teacher_trajectories_reused']==25600 and
  value['teacher_replay_excluded'] is True and value['no_head_refit'] is True and value['trajectories_regenerated'] is False and
  value['actor_optimizer_inherited'] is False and value['actor_checkpoint_inherited'] is False and value['actual_saved_head_cpu_readback'] is True,'native_initial_adoption')
 need(receipt['path']==q['initial_head_checkpoint']==str(path.with_name('initial.pt')) and
  receipt['sha256']==q['initial_head_checkpoint_sha256'] and receipt['replay_groups']==0 and receipt['completed_outer']==0 and
  receipt['head_adam_steps']==200 and receipt['actor_snapshot_id']==run['run_id']+':0','native_initial_declared_checkpoint')
 native_version(receipt['head_version'],0)
 need(provenance['teacher_replay_excluded'] is True and provenance['formal_replay_groups_at_adoption']==0 and
  provenance['warmup_head_adam_steps']==200 and provenance['no_refit'] is True and provenance['adopted_at_completed_outer']==0 and
  provenance['warmup_checkpoint_sha256']==value['raw_warmup_checkpoint_sha256']==warmup['head_sha256'] and
  warmup['source_commit']==intent['warmup_source_commit'] and warmup['source_map_sha256']==intent['warmup_source_map_sha256'] and
  warmup['ready_path']==intent['warmup_base']+'/warmup/ready.json' and
  warmup['ready_sha256']==provenance['warmup_ready_sha256'],'native_actual_warmup_lineage')
 need(proof['native_class']=='transformers.models.qwen2.modeling_qwen2.Qwen2DecoderLayer' and
  proof['official_block_copy_verified'] is True and proof['block_index']==35,'native_copied_block_producer_proof')
 return dict(status='observed',metadata_sha256=info['scanned_sha256'],initial_head_version=receipt['head_version'],
  actual_warmup_head_adam_updates=200,formal_initial_completed_outers=0,formal_initial_head_updates=0,formal_initial_replay_groups=0,
  teacher_replay_excluded=True,source_commit=warmup['source_commit'],ready_sha256=warmup['ready_sha256'],
  head_declared_sha256=hash_value(receipt['sha256']),scope='existing JSON initialization proofs; trained Head file is not read')
def recorded_loss(value):
 need(value.get('loss_mode')=='qwen_native_lastblock_binary_v1' and value.get('teacher_replay_excluded') is True,'native_saved_loss_kind')
 outer=value['completed_outer'];need(integer(outer,1),'native_saved_head_outer')
 native_version(value.get('head_version',value.get('version')),outer)
 need(value['head_adam_steps']==native_adam(outer) and value['replay_groups']==512*min(outer,8),'native_saved_head_recent8_counts')
 return dict(status='observed',loss_mode=value['loss_mode'],auxiliary_weight=.3,target_statistic='sample_std',
  reduction='mean(raw_age_weight_i*loss_i); no normalization',teacher_replay_excluded=True)
def head_commit():
 need(complete_updates is not None,'complete_prefix_unknown')
 if not complete_updates:return dict(status='missing',reason='no_recorded_complete_outer')
 latest=complete_updates[-1];outer=latest['step'];value,info=document('artifacts/question-probe/completed/'+latest['attempt_id']+'.json')
 need(value is not None,'head_commit_metadata_missing')
 need(value['stage']=='question_probe_outer_committed' and value['outer']==outer and value['attempt_id']==latest['attempt_id'] and
  value['successful_actor_snapshot_id']==run['run_id']+':'+str(outer),'native_head_actor_identity')
 native_version(value['head_version'],outer);native_version(value['prediction_version'],outer-1)
 h=value['head_update'];n=min(outer,8)
 need(h['completed_outer']==h['update_count']==outer and h['fit_epochs']==2 and h['replay_groups']==512*n and
  h['head_optimizer_updates']==4*n and h['head_adam_steps_before']==native_adam(outer-1) and
  h['head_adam_steps_after']==native_adam(outer) and h['teacher_replay_excluded'] is True,'native_head_recent8_actual_counts')
 summary=h['training_summary'];need(summary['teacher_replay_excluded'] is True and
  summary['weight_reduction']=='mean(raw_age_weight_i*loss_i); no normalization','native_raw_replay_reduction')
 total=summary['total_loss'];bce=summary['binary_bce'];mse=summary['regression_mse_normalized'];accuracy=summary['binary_accuracy']
 need(all(number(v) for v in (total,bce,mse,accuracy)) and 0<=accuracy<=1 and
  math.isclose(total,bce+.3*mse,rel_tol=1e-6,abs_tol=2e-6),'native_head_loss_parts')
 return dict(status='observed',outer=outer,actor_snapshot_id=value['successful_actor_snapshot_id'],head_version=value['head_version'],
  prediction_version=value['prediction_version'],feature_spec_sha=hash_value(value['feature_spec_sha']),metadata_sha256=info['scanned_sha256'],
  recorded_head_update=select(h,('completed_outer','update_count','head_optimizer_updates','replay_groups','head_adam_steps_before','head_adam_steps_after')),
  training_binary_accuracy=accuracy,training_replay_weighted_loss=total,weighted_binary_bce=bce,weighted_normalized_std_mse=mse,
  teacher_replay_excluded=True,replay_age_weights=[2.,1.,.5,.25,.125,.0625,.03125,.015625],
  gpu_complete_step_window_peaks=dict(status='unknown',reason='not_computed_by_this_metadata_entry'),
  scope='existing complete-outer Head JSON; no Head tensor read or additional fit/evaluation')
'''


REMOTE_MAIN=r'''
import socket,subprocess
sys.dont_write_bytecode=True;os.environ['CUDA_VISIBLE_DEVICES']=''
raw=sys.stdin.read(1024*1024+1);need(len(raw.encode())<=1024*1024,'payload_limit')
payload=json.loads(raw);intent=payload['intent'];base=Path(intent['base']);repo=base/'repo'
result=dict(role='qprobe_native_fresh_primary_metadata_readonly',run_id=payload['expected_run'],execution_id=payload['expected_execution'],
 source_commit=intent['source_commit'],base=str(base),status='unknown',metadata_only=True,read_once=True,remote_mutation=False,
 no_model_or_tensor_reads=True,no_qa_or_token_files=True,no_native_verifier=True,
 authoritative_eligibility=dict(status='unknown',reason='published_runtime_and_actual_context_free_release_not_yet_verified'))
try:
 need(type(payload['after_step']) is int and 0<=payload['after_step']<=200,'after_step')
 run=pinned_run();result['authoritative_eligibility']=dict(status='observed',published_runtime=True,actual_context_free_release_approved=True)
 result['native_initialization']=section(native_initialization)
 complete_updates=None;closed_attempts=None;driver_attempt_partial=False
 result['births']=section(births);result['reward_window']=section(reward_window)
 result['opportunity_counts']=section(opportunity_counts);result['infra_window']=section(infra_window)
 result['runtime_status']=section(runtime_status);result['scores']=section(scores);result['native_first_five']=section(native_five)
 result['latest_head_commit']=section(head_commit);result['checkpoint_head_metadata']=section(checkpoint_metadata)
 result['gpu_samples']=section(gpu_samples)
 result['gpu_complete_step_window_peaks']=dict(status='unknown',reason='not_computed_by_this_metadata_entry')
 need('torch' not in sys.modules,'unexpected_torch_import');result['status']='observed'
except (Unknown,AssertionError,KeyError,IndexError,TypeError,ValueError,OverflowError,OSError) as failure:
 result.update(status='unknown',reason=str(failure) if isinstance(failure,Unknown) else type(failure).__name__)
finally:
 signal.alarm(0);result.update(observed_unix=time.time(),aggregate_bytes_read=spent,bounded_failures=bounded_failures,metadata_files=metadata_files,
  limits=dict(aggregate_read_bytes=TOTAL_LIMIT,ledger_bytes=LEDGER_LIMIT,rank_ledger_bytes=RANK_LEDGER_LIMIT,
   process_ledger_bytes=PROCESS_LEDGER_LIMIT,output_bytes=128*1024,deadline_seconds=65,readback_recheck_included=True))
encoded=json.dumps(result,allow_nan=False)
if len(encoded.encode())>128*1024:
 result=dict(role=result['role'],run_id=result['run_id'],execution_id=result['execution_id'],source_commit=result['source_commit'],base=result['base'],
  status='unknown',reason='output_limit',metadata_only=True,read_once=True,remote_mutation=False);encoded=json.dumps(result,allow_nan=False)
print('NATIVE_PRIMARY_JSON:'+encoded,flush=True)
'''


def _literal(path,name):
    nodes=ast.parse(path.read_bytes()).body
    return next(ast.literal_eval(n.value) for n in nodes if isinstance(n,ast.Assign)
                and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))


def remote_program():
    fresh=_literal(OPS/READERS[0][0],'REMOTE_READ');milestone=_literal(OPS/READERS[1][0],'REMOTE_READ')
    support=_literal(OPS/READERS[0][0],'REMOTE_SUPPORT')
    assert support.count('TOTAL_LIMIT=576*1024*1024')==1
    support=support.replace('TOTAL_LIMIT=576*1024*1024','TOTAL_LIMIT=64*1024*1024')
    def functions(source,names):
        nodes={n.name:n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}
        return '\n\n'.join(ast.get_source_segment(source,nodes[name]) for name in names)
    reused=functions(fresh,('births','reward_window','infra_window','opportunity_counts','runtime_status','checkpoint_metadata','gpu_samples'))
    reused+='\n'+functions(milestone,('scores','native_five'))
    code=support+'\n'+reused+'\n'+REMOTE_NATIVE+'\n'+REMOTE_MAIN;ast.parse(code)
    return code


def published_support():
    def git(*args):return subprocess.check_output(['git','-C',str(OPS),*args],timeout=30)
    commit=git('rev-parse','HEAD').decode().strip();own=Path(__file__).resolve()
    assert git('show',commit+':'+str(own.relative_to(OPS)))==own.read_bytes(),'Publish exact observer first'
    for path,source,sha in READERS:
        raw=(OPS/path).read_bytes();assert hashlib.sha256(raw).hexdigest()==sha
        assert git('show',source+':'+path)==git('show',commit+':'+path)==raw
    spec=importlib.util.spec_from_file_location('published_native_primary_transport',OPS/READERS[2][0])
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module,commit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ('handle','runtime','receipt'):parser.add_argument('--'+flag,type=Path,required=flag!='receipt')
    parser.add_argument('--expected-run',required=True);parser.add_argument('--expected-execution',required=True)
    parser.add_argument('--after-step',required=True,type=int);args=parser.parse_args()
    assert 0<=args.after_step<=200 and all(re.fullmatch('[0-9a-f]{32}',x) for x in (args.expected_run,args.expected_execution))
    flex,commit=published_support();args.handle.relative_to(ROOT/'preparation');intent=flex.read(args.handle)
    assert intent['role']=='a800_qwen_native_formal_prospective_intent' and intent['host']=='baai-a800'
    assert intent['native_formal_200_authorized'] is True and intent['target_updates']==200
    module=flex.runtime_module(args.runtime,intent);code=remote_program()
    route=module.ssh_command(code);assert route[-1].startswith('python3 -c ')
    route[-1]='/data/search-g1/env/bin/python -c '+route[-1][len('python3 -c '):]
    receipt=args.receipt or ROOT/'preparation'/('qprobe-qwen-native-primary-'+str(time.time_ns())+'.json')
    receipt.relative_to(ROOT/'preparation');assert receipt.is_absolute() and receipt.resolve()==receipt and receipt.suffix=='.json' and not receipt.exists()
    result=dict(role='single_native_qwen_primary_metadata_receipt',operator_commit=commit,run_id=args.expected_run,execution_id=args.expected_execution,
        source_commit=intent['source_commit'],base=intent['base'],after_complete_step=args.after_step,
        handle_sha256=hashlib.sha256(args.handle.read_bytes()).hexdigest(),remote_reader_sha256=hashlib.sha256(code.encode()).hexdigest(),
        published_reader_sources=[dict(path=p,commit=c,sha256=s) for p,c,s in READERS],no_automatic_retry=True,observation=None)
    payload=json.dumps(dict(intent=intent,expected_run=args.expected_run,expected_execution=args.expected_execution,after_step=args.after_step),allow_nan=False).encode()
    with tempfile.TemporaryFile() as out,tempfile.TemporaryFile() as err:
        try:
            done=subprocess.run(route,input=payload,stdout=out,stderr=err,timeout=90);result.update(exit_code=done.returncode,status='READ_RETURNED')
        except (OSError,subprocess.TimeoutExpired) as failure:result.update(exit_code=None,status='TRANSPORT_UNKNOWN',failure_type=type(failure).__name__)
        result.update(stdout_bytes=out.tell(),stderr_bytes=err.tell());out.seek(0);raw=out.read(128*1024+1)
        err.seek(0);result['bounded_stderr_sha256']=hashlib.sha256(err.read(64*1024)).hexdigest()
    if result['exit_code']==0 and len(raw)<=128*1024:
        try:
            rows=[json.loads(line.split(':',1)[1]) for line in raw.decode().splitlines() if line.startswith('NATIVE_PRIMARY_JSON:')]
            assert len(rows)==1;value=rows[0]
            assert (value['run_id'],value['execution_id'],value['source_commit'],value['base'])==(args.expected_run,args.expected_execution,intent['source_commit'],intent['base'])
            assert value['metadata_only'] is True and value['read_once'] is True and value['remote_mutation'] is False
            result['observation']=value
        except (AssertionError,KeyError,TypeError,ValueError,UnicodeError) as failure:result.update(status='SANITIZED_OUTPUT_REJECTED',failure_type=type(failure).__name__)
    elif len(raw)>128*1024:result['status']='OUTPUT_LIMIT_EXCEEDED'
    result['observed_unix']=time.time();flex.once(receipt,result)
    print(json.dumps(dict(receipt=str(receipt),**result),allow_nan=False))
    return 0 if result['observation'] is not None and result['observation']['status']=='observed' else 1


if __name__=='__main__':raise SystemExit(main())
