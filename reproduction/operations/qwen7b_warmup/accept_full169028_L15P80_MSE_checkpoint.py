"""Terminal-only full169028 L15/P80/MSE CPU acceptance; stage never loads CP.

Both explicit training seeds are separate runroots. The shared8GiB mutex guards
all model loads. Physical execution/source and fullP20 alignment gates remain
required; this CPU proof does not claim a CUDA next-update restore test.
"""
import argparse,fcntl,hashlib,importlib,inspect,json,math,os,pathlib,random,resource,subprocess,time
A=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021')
COMMON_LOCK=A/'online-mixed-P80-handoff-CPU-acceptance-20261023-v1/acceptance.lock'
OUT=A/'full169028-L15P80-MSE-terminal-CPU-acceptance-20261023-v1'
FIXTURE=A/'full169028-policy80-physical-ownG10-20261023-v2/fixture'
PARENTS={
 'full169028-L15P80-MSE-ownG5-physical-seed9137-plan2026102227-20261023-v3':(9137,2026102227),
 'full169028-L15P80-MSE-ownG5-physical-seed19137-plan2026102302-20261023-v3':(19137,2026102302),
}
MODULE='reproduction.operations.qwen7b_warmup.online_packed_full169028_MSE_P80'
def read(p):return json.loads(pathlib.Path(p).read_bytes())
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def proof(p):
 p=pathlib.Path(p);h=hashlib.sha256()
 with p.open('rb')as f:
  for chunk in iter(lambda:f.read(8388608),b''):h.update(chunk)
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def checked(ref):
 p=pathlib.Path(ref['path']);assert proof(p)==ref;return p
def atomic(p,v):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(v,sort_keys=True,allow_nan=False)+'\n');q.replace(p)
def recipe_for(base,binding):
 init,plan=PARENTS[base.name];assert base.parent==A
 expected=dict(architecture='one_question_packed_matched_objective_v1',training_scope='full169028_fresh_official_actor_not_pilot_resume',objective='mse',fixed_m=5,layer=15,capture_layer=14,prefix=80,head_max_positions=2048,EMA_decay=.9995,lr=5e-6,global_batch=128,training_questions=169028,seed=plan,init_seed=init,train_plan_seed=plan,diagnostic_seed=2026102228,seed_source='explicit_CLI',configured_local_batch=64,configured_micro=8,configured_accumulation=8,tail_local_batch=34,tail_global_batch=68,updates_per_epoch=1321,class_weighting=False,sampling_independent_of_EM=True,question_prompt_occurrences_per_group=1)
 assert all(binding.get(k)==v for k,v in expected.items())and 'cache_window_capacity'not in binding
 assert binding['fixture']['path']==str(FIXTURE/'fixture-complete.json')
 head=binding['initial_head_sha256'];assert isinstance(head,str)and len(head)==64 and all(c in'0123456789abcdef'for c in head)
 return dict(init_seed=init,train_plan_seed=plan,diagnostic_seed=2026102228,initial_head_sha256=head,module=MODULE)
def verify_stamps(value,recipe,head=False):
 assert all(value[k]==recipe[k]for k in('init_seed','train_plan_seed','diagnostic_seed'))
 if head:assert value['initial_head_sha256']==recipe['initial_head_sha256']
def verify_saved_phase(blob,epoch,recipe):
 verify_stamps(blob,recipe,head=True);assert blob['epoch']==epoch and blob['cursor']==blob['batch_attempts']==blob['actual_Adam_updates']==blob['EMA_updates']==epoch*1321
 assert blob['margin_satisfied_batches']==0 and blob['actor_updates']==0 and blob['epoch_complete']is True and blob['next_epoch']==epoch+1 and blob['within_next_epoch_question_position']==0 and blob['execution_phase']=='epoch_complete_pre_evaluation'
 assert len(blob['RNG_by_rank'])==2 and all(set(x)=={'python','numpy','torch','cuda'}for x in blob['RNG_by_rank'])
 assert all(k in blob for k in('raw','EMA','Adam'))
def new_sampling_diagnostic():
 return {k:dict(questions=0,G5_all_wrong=0,G5_Mixed=0,G5_all_correct=0)for k in range(11)}
def add_sampling_question(counts,all_labels,selected_labels):
 k=sum(all_labels);j=sum(selected_labels);counts[k]['questions']+=1;counts[k]['G5_all_wrong'if j==0 else'G5_all_correct'if j==5 else'G5_Mixed']+=1
def finish_sampling_diagnostic(counts):
 denom=math.comb(10,5);rows={};observed_correct=observed_wrong=expected_correct_num=expected_wrong_num=questions=0
 for k,c in counts.items():
  n=c['questions'];correct=math.comb(k,5)if k>=5 else 0;wrong=math.comb(10-k,5)if 10-k>=5 else 0;row=dict(c,K10=k,G10_Mixed=0<k<10,probability_G5_all_correct_numerator_over_252=correct,probability_G5_all_wrong_numerator_over_252=wrong,probability_G5_homogeneous_numerator_over_252=correct+wrong,expected_G5_all_correct=n*correct/denom,expected_G5_all_wrong=n*wrong/denom,expected_G5_homogeneous=n*(correct+wrong)/denom);rows[str(k)]=row
  if 0<k<10:questions+=n;observed_correct+=c['G5_all_correct'];observed_wrong+=c['G5_all_wrong'];expected_correct_num+=n*correct;expected_wrong_num+=n*wrong
 return dict(status='ACTUAL_PLAN_G10_MIXED_TO_G5_HOMOGENEOUS_VS_EXACT_HYPERGEOMETRIC_EXPECTATION',K10_strata=rows,G10_Mixed_questions=questions,observed_G10Mixed_to_G5_all_correct=observed_correct,observed_G10Mixed_to_G5_all_wrong=observed_wrong,observed_G10Mixed_to_G5_homogeneous=observed_correct+observed_wrong,expected_G10Mixed_to_G5_all_correct_numerator=expected_correct_num,expected_G10Mixed_to_G5_all_wrong_numerator=expected_wrong_num,expectation_denominator=denom,expected_G10Mixed_to_G5_homogeneous=(expected_correct_num+expected_wrong_num)/denom,expectation_formula='[C(K10,5)+C(10-K10,5)]/C(10,5)',source='Already reconstructed and matched actual saved plan; no extra RAW scan or plan generation',assumption='Natural label-independent uniform5-of10 without replacement; expected counts are not enforced class balancing')
def terminal_gate(base):
 if not(base/'complete.json').is_file()or not(base/'root-submission.json').is_file():return dict(status='WAITING_ACTUAL_TERMINAL_CHECKPOINT_NOT_OPENED')
 submission=read(base/'root-submission.json');job=str(submission['job_id']);accounting=subprocess.check_output(['sacct','-X','-n','-P','-j',job,'--format=JobIDRaw,State,ExitCode'],universal_newlines=True,timeout=15)
 if not any(row.split('|')[:3]==[job,'COMPLETED','0:0']for row in accounting.splitlines()):return dict(status='WAITING_ACTUAL_TERMINAL_CHECKPOINT_NOT_OPENED',job_id=job,accounting=accounting.strip())
 return None
def finite(t,torch):
 assert torch.is_tensor(t)and t.device.type=='cpu'and t.layout==torch.strided
 if t.is_floating_point():
  flat=t.detach().reshape(-1)
  for i in range(0,flat.numel(),1048576):assert bool(torch.isfinite(flat[i:i+1048576]).all())
 return t.numel()
def tensor_mapping_sha(mapping,torch):
 h=hashlib.sha256()
 for name,t in mapping.items():
  h.update(name.encode());h.update(str(t.dtype).encode());h.update(repr(tuple(t.shape)).encode());raw=t.detach().contiguous().reshape(-1).view(torch.uint8)
  for i in range(0,raw.numel(),1048576):h.update(raw[i:i+1048576].numpy().tobytes())
 return h.hexdigest()
def serialized_cuda_rng_bytes(t,torch):
 if not(torch.is_tensor(t)and t.device.type=='cpu'and t.dtype==torch.uint8 and t.ndim==1 and t.numel()>=16):raise ValueError('Expected serialized CUDA RNG state: CPU uint8 vector with at least16bytes')
 return t.numel()
def validate(base,epoch,out):
 gate=terminal_gate(base)
 if gate:return gate
 import numpy as np
 import torch
 from transformers import Qwen2Config
 from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer,Qwen2RotaryEmbedding
 from reproduction.operations.qwen7b_warmup.packed_variance_probe import PackedVarianceHead,PackedVarianceProbe
 runtime=importlib.import_module(MODULE);started=time.time();submission=read(base/'root-submission.json');job=str(submission['job_id']);complete=read(base/'complete.json');controller=read(base/'controller.json');binding=complete['binding'];recipe=recipe_for(base,binding)
 assert complete['status']=='FULL169028_P80_OWNG5_MSE_TRAINING_COMPLETE'and complete['epochs']==10 and complete['training_questions']==169028 and complete['cursor']==complete['batch_attempts']==complete['actual_Adam_updates']==complete['EMA_updates']==13210 and complete['margin_satisfied_batches']==0 and not complete['final4096_scored']
 assert all(controller.get(k)==v for k,v in binding.items())and str(controller['job_id'])==job
 pub=read(base/'source-publication.json');assert pub['commit']==submission['source_commit']
 for rel,ref in pub['files'].items():assert proof(pathlib.Path(pub['code_root'])/rel)==dict(path=str(pathlib.Path(pub['code_root'])/rel),**ref)
 ready=read(base/'ready.json');assert ready['publication']==proof(base/'source-publication.json')and ready['fixture']==binding['fixture']and ready['training_questions']==169028 and ready['own_G10_trajectories']==1690280
 fixture_path=checked(binding['fixture']);fixture=read(fixture_path);assert fixture['status']==runtime.FIXTURE_STATUS and fixture['training_questions']==169028 and fixture['max_real_policy_prefix']==80
 accepted_path=checked(ready['independent_acceptance']);accepted=read(accepted_path);assert str(accepted['status']).endswith('ACCEPTED')and accepted['fixture']==binding['fixture']
 physical_ref=runtime.full_acceptance_physical_source_ref(accepted);assert physical_ref==binding['physical_source_contract']==ready['physical_source_contract'];runtime.validate_physical_source_contract(read(checked(physical_ref)))
 alignment_ref=runtime.full_acceptance_alignment_ref(accepted);assert alignment_ref==ready['P20_alignment'];checked(alignment_ref);assert ready['fresh_full_P20_prefix_prompt_and_own_EM_exact']is True and ready['actual_physical_execution_and_seal_source_exact']is True
 assert fixture['validation_groups']['sha256']=='50f1e3a6927d02c0f8cadf1feb5b4229b075b5d099fe31fd0de5c2ae3b149292'and fixture['fixed_train_G5']['sha256']=='e6e74d92160efb0740ba5c0316c4ee7686f06a499c90cafd5a545f325b62209b'
 checked(fixture['validation_groups']);checked(fixture['fixed_train_G5']);groups=read(checked(fixture['train_groups']));assert len(groups)==169028 and all(len(g['trajectory_ids'])==len(g['labels'])==10 for g in groups.values())
 plans={};sampling_diagnostics={}
 for e in range(1,11):
  plan=runtime.epoch_plan(groups,e,5,train_plan_seed=recipe['train_plan_seed']);sampling_counts=new_sampling_diagnostic();coverage=read(base/('coverage-epoch-%02d.json'%e));verify_stamps(coverage,recipe);saved=read(base/('sampling-plan-epoch-%02d.json'%e));verify_stamps(saved,recipe);ps=digest(plan)
  assert coverage['questions']==coverage['unique_questions']==169028 and coverage['sampling_independent_of_EM']and coverage['trajectory_visits']==845140 and coverage['plan_sha256']==saved['plan_sha256']==ps and saved['rows']==plan
  assert len({v['qid']for v in plan})==169028 and all(v['M']==5 for v in plan)
  for view in plan:
   g=groups[view['qid']];add_sampling_question(sampling_counts,g['labels'],view['labels']);own=dict(zip(g['trajectory_ids'],g['labels']));assert view['labels']==[own[t]for t in view['trajectory_ids']]and view['true_mixed']==(0<sum(view['labels'])<5);mean=sum(view['labels'])/5;assert view['target_variance']==mean*(1-mean)and view['target4variance']==4*view['target_variance']
  plans[e]=ps;sampling_diagnostics[str(e)]=dict(plan_sha256=ps,**finish_sampling_diagnostic(sampling_counts))
 assert len(set(plans.values()))==10;del plan,saved
 path=base/('checkpoint-epoch-%02d.pt'%epoch);before=path.stat();fresh=proof(path)
 if epoch==10:assert read(base/'latest.json')==fresh
 atomic(out/'progress.json',dict(stage='CPU_MMAP_STRICT_FULL169028_RESTORE',checkpoint=fresh,time=time.time()))
 blob=torch.load(path,map_location='cpu',mmap=True,weights_only=False);assert blob['binding']==binding;verify_saved_phase(blob,epoch,recipe);assert blob['plan_sha256']==plans[epoch];steps=blob['actual_Adam_updates']
 config=Qwen2Config.from_pretrained(str(A.parent/'assets/qwen25-7b-instruct-a09a35458c70'),local_files_only=True);config._attn_implementation='flash_attention_2'
 with torch.device('meta'):
  block=Qwen2DecoderLayer(config,layer_idx=15).float();rotary=Qwen2RotaryEmbedding(config=config).float();head=PackedVarianceHead(max_positions=2048);probe=PackedVarianceProbe(block,rotary,'cache_position'in inspect.signature(block.forward).parameters,head)
 geometry={n:tuple(t.shape)for n,t in probe.state_dict().items()};assert geometry['group_head.positions.weight']==(2048,256)
 models={}
 for state in ('raw','EMA'):
  mapping=blob[state];assert set(mapping)==set(geometry)and all(tuple(mapping[n].shape)==s for n,s in geometry.items());elements=sum(finite(t,torch)for t in mapping.values());probe.load_state_dict(mapping,strict=True,assign=True)
  assert all(torch.equal(t,mapping[n])for n,t in probe.state_dict().items());models[state]=dict(tensors=len(mapping),elements=elements,strict_load=True,sha256=tensor_mapping_sha(mapping,torch))
 assert models['raw']['sha256']!=models['EMA']['sha256'];probe.load_state_dict(blob['raw'],strict=True,assign=True);opt=torch.optim.Adam(probe.parameters(),lr=5e-6);opt.load_state_dict(blob['Adam']);ids=[i for group in blob['Adam']['param_groups']for i in group['params']];params=list(probe.parameters());assert len(ids)==len(params)and set(ids)==set(blob['Adam']['state'])
 for sid,param in zip(ids,params):
  saved=blob['Adam']['state'][sid];loaded=opt.state[param];assert int(saved['step'])==int(loaded['step'])==steps
  for name in ('exp_avg','exp_avg_sq'):
   assert saved[name].shape==param.shape and torch.equal(saved[name],loaded[name]);finite(saved[name],torch)
  assert bool(saved['exp_avg_sq'].ge(0).all())
 assert all(g['lr']==5e-6 for g in opt.param_groups)
 rngs=[];assert len(blob['RNG_by_rank'])==2
 for rank,rng in enumerate(blob['RNG_by_rank']):
  assert set(rng)=={'python','numpy','torch','cuda'};random.Random().setstate(rng['python']);np.random.RandomState().set_state(rng['numpy']);torch.Generator(device='cpu').set_state(rng['torch']);rngs.append(dict(rank=rank,CPU_states_restored=True,CUDA_serialized_bytes=serialized_cuda_rng_bytes(rng['cuda'],torch),CUDA_runtime_restore_tested=False))
 evaluation=read(base/('evaluation-epoch-%02d.json'%epoch));assert evaluation['cursor']==blob['cursor']and evaluation['binding']==binding and not evaluation['final4096_scored']
 after=path.stat();assert(before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
 result=dict(status='TERMINAL_FULL169028_P80_MSE_CPU_SHA_MODEL_ADAM_RNG_PLAN_PHYSICAL_GATE_ACCEPTED',time=time.time(),job_id=job,checkpoint=fresh,epoch=epoch,batch_attempts=blob['cursor'],actual_Adam_updates=steps,EMA_updates=blob['EMA_updates'],epoch_complete=True,next_epoch=blob['next_epoch'],within_next_epoch_question_position=0,execution_phase=blob['execution_phase'],binding=binding,explicit_seed_recipe=recipe,models=models,Adam_parameter_states=len(ids),Adam_CPU_restore=True,RNG_by_rank=rngs,epoch_plan_SHAs=plans,source_publication=proof(base/'source-publication.json'),source_complete=proof(base/'complete.json'),source_evaluation=proof(base/('evaluation-epoch-%02d.json'%epoch)),independent_fixture_acceptance=proof(accepted_path),accepted_input=dict(fixture=binding['fixture'],physical_source_contract=physical_ref,P20_alignment=alignment_ref,independent_acceptance=proof(accepted_path)),actual_physical_execution_source_contract=physical_ref,full_P20_alignment=alignment_ref,goal6_natural_G5_hard_negative_diagnostic=sampling_diagnostics,CPU_only=True,GPU_forward_or_next_update_restore_tested=False,final4096_read=False,shared_global_CPU_lock=str(COMMON_LOCK),seconds=time.time()-started,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
 atomic(out/'accepted.json',result);return result
def main():
 p=argparse.ArgumentParser();p.add_argument('--base',type=pathlib.Path,required=True);p.add_argument('--epoch',type=int,default=10);p.add_argument('--source-sha',required=True);a=p.parse_args();assert os.uname().sysname=='Linux'and os.environ.get('CUDA_VISIBLE_DEVICES')==''and 1<=a.epoch<=10 and proof(pathlib.Path(__file__))['sha256']==a.source_sha and a.base.name in PARENTS and a.base.parent==A
 out=OUT/a.base.name/('epoch-%02d'%a.epoch);out.mkdir(parents=True,exist_ok=True);gate=terminal_gate(a.base)
 if gate:atomic(out/'run-complete.json',gate);print(json.dumps(gate));return
 with COMMON_LOCK.open('a+')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);atomic(out/'controller.json',dict(pid=os.getpid(),time=time.time(),source_sha=a.source_sha));result=validate(a.base,a.epoch,out);atomic(out/'run-complete.json',result);print(json.dumps(dict(status=result['status'],receipt=str(out/'accepted.json'))))
if __name__=='__main__':main()
