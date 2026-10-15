"""Remote-only, read-only native Head200 controls and fresh512 diagnostic.

Run this published standalone operator against an unmodified pinned runtime.
It never fits a head or updates an actor. Raw labels, tokens and predictions
stay in the explicit remote diagnostic directory.
"""
import argparse
import copy
import csv
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import socket
import time
import traceback
import uuid

sys.dont_write_bytecode=True
RUNTIME_SHA='fcb77204f48d5cb8ec21c68df7d4a2c937b3557a'
WARMUP='/data/shaoshitong/Search-G1-qprobe-qwen-lastblock-warmup10-096c8297cba74f8089063e763686671e/warmup'
HEAD_SHA='9b32f40ea573b855dc9e3be627fce7be1c03a1ba0b0e374808a81db311bb6849'
TEACHER='/data/shaoshitong/Search-G1-qprobe-shared/teacher-warmup-v4-sigma3-fullouter'
ALLOWED=(0,1,2,3,5,6)


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(8*1024**2),b''):h.update(part)
    return h.hexdigest()


def remote(path):
    p=Path(path)
    assert p.is_absolute() and p.resolve()==p and str(p).startswith(('/data/','/mnt/','/backup/'))
    return p


def once(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:
        json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def bind(args):
    runtime=remote(args.runtime_repo)
    assert subprocess.check_output(['git','-C',str(runtime),'rev-parse','HEAD'],text=True).strip()==RUNTIME_SHA
    assert not subprocess.check_output(['git','-C',str(runtime),'status','--porcelain']).strip(), 'Diagnostic runtime must remain pinned and clean'
    sys.path.insert(0,str(runtime))
    out=remote(args.output_dir)
    assert not out.is_relative_to(runtime) and not out.is_relative_to(Path(args.formal_base))
    out.mkdir(parents=True,exist_ok=True)
    for name in ('cache/hf','cache/triton','cache/inductor','tmp'):(out/name).mkdir(parents=True,exist_ok=True)
    os.environ.update(HF_HOME=str(out/'cache/hf'),XDG_CACHE_HOME=str(out/'cache'),TMPDIR=str(out/'tmp'),
        TRITON_CACHE_DIR=str(out/'cache/triton'),TORCHINDUCTOR_CACHE_DIR=str(out/'cache/inductor'),
        HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1')
    return runtime,out


def heads(args):
    import torch
    warmup=remote(args.warmup_dir);ready=read(warmup/'ready.json');path=warmup/'fresh-head.pt'
    assert ready['head_sha256']==HEAD_SHA==sha(path) and ready['head_adam_steps']==200
    assert ready['stage']=='QWEN_LAST_BLOCK_FRESH_HEAD_WARMUP10_COMPLETE' and ready['global_max_tokens']==244
    raw=torch.load(path,map_location='cpu',weights_only=False)
    head=raw['head'];assert head['fitted'] is True and head['update_count']==200
    run=read(remote(args.formal_base)/'logs/run-manifest.json')
    initial=remote(run['question_probe']['initial_head_checkpoint'])
    initial_sha=run['question_probe']['initial_head_checkpoint_sha256']
    assert initial_sha.startswith('014c') and sha(initial)==initial_sha, 'Only the original formal initial head is allowed'
    formal=torch.load(initial,map_location='cpu',weights_only=False)
    assert formal['completed_outer']==0 and formal['head_adam_updates']==200
    return ready,raw,head,formal,dict(warmup_checkpoint=str(path),warmup_sha256=HEAD_SHA,
        formal_initial_checkpoint=str(initial),formal_initial_sha256=initial_sha,
        warmup_ready_sha256=sha(warmup/'ready.json'),formal_source_commit=run['source_commit'])


def compare_heads(warm,formal,identity):
    import torch
    a,b=warm['model'],formal['model'];assert set(a)==set(b)
    rows={}
    for key in sorted(a):
        x,y=a[key],b[key]
        shape=x.shape==y.shape;dtype=x.dtype==y.dtype
        equal=shape and dtype and torch.equal(x,y)
        delta=float((x.double()-y.double()).abs().max()) if shape and x.numel() else 0.
        rows[key]=dict(shape=list(x.shape),formal_shape=list(y.shape),dtype=str(x.dtype),formal_dtype=str(y.dtype),
            exact_equal=equal,max_abs_difference=delta,
            warmup_tensor_sha256=hashlib.sha256(x.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
            formal_tensor_sha256=hashlib.sha256(y.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest())
    return dict(role='original_warmup200_vs_formal_initial_tensor_semantics',**identity,
        tensor_count=len(rows),all_model_tensors_equal=all(x['exact_equal'] for x in rows.values()),tensors=rows,
        warmup_adam_updates=warm['update_count'],formal_head_adam_updates=formal['head_adam_updates'],
        formal_completed_outer=formal['completed_outer'],model_state_only=True,
        checkpoint_wrapper_and_replay_metadata_may_differ=True,checked_unix=time.time())


def warmup_groups(raw):
    from search_g1.qprobe_warmup import moments
    root=remote(TEACHER);header=read(root/'identity.json');complete=read(root/'complete.json')
    provenance=raw['label_producer_provenance']
    assert sha(root/'identity.json')==provenance['teacher_identity_sha256']
    assert sha(root/'complete.json')==provenance['teacher_complete_sha256']
    groups=[]
    for item in complete['chunks']:
        p=root/item['relative_path'];assert p.parent==root and sha(p)==item['sha256']
        groups.extend(read(p)['groups'])
    assert len(groups)==5120 and len({g['qid'] for g in groups})==5120
    for binding,g in zip(header['questions'],groups):
        assert all(g[k]==binding[k] for k in ('qid','question_hash'))
        assert (g['correct_count'],g['mean'],g['std'])==moments(g['strict_em'])
    return groups


def model_for(head,device):
    import dataclasses
    import torch
    from search_g1.qprobe_qwen_block_model import QwenBlockQProbeConfig,QwenLastBlockQuestionProbe
    names={f.name for f in dataclasses.fields(QwenBlockQProbeConfig)}
    config=QwenBlockQProbeConfig(**{k:v for k,v in head['config'].items() if k in names})
    assert config.micro_batch_size==32 and config.max_tokens==244 and config.auxiliary_weight==.3
    with torch.random.fork_rng(devices=[]):model=QwenLastBlockQuestionProbe(config)
    model.load_state_dict(head['model'],strict=True);model.requires_grad_(False)
    model.to(device=device,dtype=torch.float32);model.eval()
    return model,config


def evaluate(head,groups,cache,out,prefix,device,*,saved_predictions=None,compare_model_state=None):
    import torch
    from search_g1.qprobe_token_model import TokenBinaryWarmup,token_binary_loss
    from search_g1.qprobe_qwen_block_model import QwenBlockWarmup
    qids=[g['qid'] for g in groups];assert qids and len(set(qids))==len(qids)
    counts=torch.tensor([g['correct_count'] for g in groups],dtype=torch.long)
    assert all(cache.index[g['qid']]['question_hash']==g['question_hash'] for g in groups)
    model,config=model_for(head,device)
    batcher=QwenBlockWarmup.__new__(QwenBlockWarmup);batcher.config=config;batcher.device=torch.device(device)
    lengths=torch.tensor([cache.index[qid]['length'] for qid in qids],dtype=torch.long)
    totals=dict(total_loss=0.,binary_bce=0.,regression_mse_raw=0.,regression_mse_normalized=0.,regression_contribution=0.)
    predictions=[];features_reencoded=0
    comparison=None
    if compare_model_state is not None:
        comparison,_=model_for(compare_model_state,device)
    max_prediction_delta=0.
    with torch.inference_mode():
        for index in torch.arange(len(qids)).split(32):
            features,sizes=TokenBinaryWarmup._batch(batcher,cache,index,lengths,qids)
            pred=model(features,sizes);loss,parts=token_binary_loss(pred,counts[index],config)
            for key,value in dict(total_loss=loss,**parts).items():totals[key]+=float(value)*len(index)
            if comparison is not None:
                check=comparison(features,sizes)
                max_prediction_delta=max(max_prediction_delta,*[float((pred[k]-check[k]).abs().max()) for k in ('binary_logit','mixed_probability','predicted_statistic')])
            for j,ordinal in enumerate(index.tolist()):
                g=groups[ordinal];p=float(pred['mixed_probability'][j]);logit=float(pred['binary_logit'][j]);std=float(pred['predicted_statistic'][j])
                row=dict(qid=g['qid'],question_hash=g['question_hash'],correct_count=g['correct_count'],
                    strict_em=g['strict_em'],true_mixed=0<g['correct_count']<5,
                    binary_logit=logit,mixed_probability=p,predicted_sample_std=std,predicted_mixed=p>=.5)
                if saved_predictions is not None:
                    old=saved_predictions[g['qid']];row['formal_saved_probability']=old
                    row['saved_probability_abs_difference']=abs(p-old)
                predictions.append(row)
            print(json.dumps(dict(stage=prefix+'_inference_progress',completed=len(predictions),total=len(qids))),flush=True)
    tp=sum(r['true_mixed'] and r['predicted_mixed'] for r in predictions)
    tn=sum(not r['true_mixed'] and not r['predicted_mixed'] for r in predictions)
    fp=sum(not r['true_mixed'] and r['predicted_mixed'] for r in predictions)
    fn=sum(r['true_mixed'] and not r['predicted_mixed'] for r in predictions)
    probability=[r['mixed_probability'] for r in predictions];logits=[r['binary_logit'] for r in predictions]
    summary=dict(role=prefix,questions=len(qids),correct=tp+tn,binary_accuracy=(tp+tn)/len(qids),
        binary_confusion=dict(true_positive=tp,true_negative=tn,false_positive=fp,false_negative=fn),
        true_mixed_fraction=sum(r['true_mixed'] for r in predictions)/len(qids),
        predicted_mixed_fraction=sum(r['predicted_mixed'] for r in predictions)/len(qids),
        correct_count_histogram={str(k):sum(r['correct_count']==k for r in predictions) for k in range(6)},
        binary_logit_range=[min(logits),max(logits)],mixed_probability_range=[min(probability),max(probability)],
        probability_exact_zero_count=probability.count(0.),probability_exact_one_count=probability.count(1.),
        threshold=.5,threshold_tie_count=probability.count(.5),**{k:v/len(qids) for k,v in totals.items()},
        model_eval=True,compute_dtype='float32',micro_batch_size=32,all_real_tokens=True,
        padding='original native forward: active micro max then flatten zero-pad244',
        loss_identity=config.loss_identity,features_reencoded=features_reencoded,
        actor_optimizer_updates=0,head_optimizer_updates=0,cache_dir=str(cache.path),
        cache_spec_sha256=cache.spec_sha,cache_manifest_sha256=sha(cache.path/'manifest.json'),checked_unix=time.time())
    if comparison is not None:summary['formal_initial_same_input_prediction_max_abs_difference']=max_prediction_delta
    if saved_predictions is not None:
        diffs=[r['saved_probability_abs_difference'] for r in predictions]
        summary.update(saved_probability_max_abs_difference=max(diffs),saved_probability_mean_abs_difference=sum(diffs)/len(diffs),
            saved_threshold_class_disagreements=sum((r['formal_saved_probability']>=.5)!=r['predicted_mixed'] for r in predictions))
    once(out/(prefix+'-predictions.json'),predictions);once(out/(prefix+'-summary.json'),summary)
    del model,comparison;torch.cuda.empty_cache()
    return summary


def replay_control(args,out):
    from search_g1.qprobe_token_features import TokenFeatureCache
    ready,raw,head,formal,identity=heads(args)
    cache=TokenFeatureCache(ready['feature_cache_dir'],expected_spec_sha=ready['token_feature_spec_sha256'])
    groups=warmup_groups(raw)
    replay=evaluate(head,groups,cache,out,'warmup5120-replay','cuda:0')
    replay['expected_original_correct']=5114;replay['reproduces_original_5114_of_5120']=replay['correct']==5114
    once(out/'warmup5120-reproduction.json',replay)
    attempts=remote(args.formal_base)/'artifacts/question-probe/attempts'
    candidates=[]
    for p in sorted(attempts.glob('*/selection.json')):
        selection=read(p)
        if selection['step']==1 and (p.parent/'rollout-audit.json').exists():candidates.append((p,selection))
    assert candidates, 'No completed formal1 actual selected512 controls available'
    path,selection=candidates[0];audit=read(path.parent/'rollout-audit.json')
    selected={g['qid']:g for g in selection['questions'] if g['selected']}
    assert len(selected)==512
    groups=[dict(qid=g['question_id'],question_hash=selected[g['question_id']]['question_hash'],
        strict_em=g['strict_correctness'],correct_count=sum(g['strict_correctness'])) for g in audit['groups']]
    run=read(remote(args.formal_base)/'logs/run-manifest.json');full=TokenFeatureCache(run['question_probe']['cache_dir'])
    measured=evaluate(head,groups,full,out,'formal1-selected512-control','cuda:0',
        saved_predictions={qid:g['predicted_mixed_probability'] for qid,g in selected.items()},compare_model_state=formal)
    result=dict(role='HEAD200_FAST_CONTROL_COMPLETE',**identity,tensor_comparison_performed_here=False,
        warmup_replay=replay,formal1_selected_control=measured,formal1_selection_sha256=sha(path),
        formal1_rollout_audit_sha256=sha(path.parent/'rollout-audit.json'),new_random512_test=False,
        head_trained=False,actor_trained=False,checked_unix=time.time())
    once(out/'replay-control-complete.json',result);print(json.dumps(result),flush=True)


def infer(args,out):
    from search_g1.qprobe_token_features import TokenFeatureCache
    from search_g1.qprobe_warmup import moments,verify_frozen
    ready,raw,head,formal,identity=heads(args)
    labels=read(remote(args.labels));groups=labels['groups']
    assert len(groups)==512 and labels['role']=='actual_frozen_official_new512_G5_labels'
    assert labels['questions']==512 and labels['trajectories']==2560
    selection_path=remote(args.labels).parent.parent/'selection.json'
    selection=read(selection_path);assert sha(selection_path)==labels['selection_sha256']
    assert [(r['qid'],r['question_hash']) for r in selection['questions']]==[(g['qid'],g['question_hash']) for g in groups]
    for g in groups:assert moments(g['strict_em'])[0]==g['correct_count']
    verify_frozen(labels['frozen_workers']['before'],labels['frozen_workers']['after'],4)
    run=read(remote(args.formal_base)/'logs/run-manifest.json')
    cache=TokenFeatureCache(run['question_probe']['cache_dir'])
    result=evaluate(head,groups,cache,out,'fresh512-independent', 'cuda:0',compare_model_state=formal)
    result.update(**identity,labels_sha256=sha(args.labels),new_random512_test=True)
    once(out/'new512-complete.json',result);print(json.dumps(result),flush=True)


def stage(args,out):
    """Select an unbiased fresh subset and prepare only remote diagnostic inputs."""
    import numpy as np
    import pyarrow.parquet as pq
    from search_g1.qprobe_features import original_question_and_id,question_text_sha
    from search_g1.qprobe_token_features import TokenFeatureCache
    from search_g1.mixed_selection import question_identity
    from search_g1.qprobe_selection import QuestionPool
    source=remote(args.formal_base);repo=remote(args.runtime_repo)
    assert repo==out/'repo', 'Teacher requires its own real pinned repository under the new base'
    assert out.parent==Path('/data/shaoshitong') and out.name.startswith('Search-G1-a800-fresh200-')
    assert not (out/'selection.json').exists() and not (out/'logs/run-manifest.json').exists()
    original=read(source/'logs/run-manifest.json');assert original['source_commit']==RUNTIME_SHA
    assert original['initial_model']==original['reference_model']=='/data/search-g1/assets/qwen2.5-3b-instruct'
    assert not original.get('full_state_restore') and original['training_world_size']==4
    teacher=read(remote(TEACHER)/'identity.json');warm_ids={g['qid'] for g in teacher['questions']}
    assert len(warm_ids)==5120
    exclusions=set(warm_ids);receipts=[]
    excluded_hashes={g['question_hash'] for g in teacher['questions']};excluded_contents=set()
    for path in sorted((source/'artifacts/question-probe/attempts').glob('*/selection.json')):
        record=read(path);exclusions.update(g['qid'] for g in record['questions'])
        excluded_hashes.update(g['question_hash'] for g in record['questions'])
        excluded_contents.update(g['content_sha256'] for g in record['questions'])
        receipts.append(dict(path=str(path),sha256=sha(path),candidates=len(record['questions']),step=record['step']))
    train=source/'data/train.parquet';assert sha(train)==original['input_sha256']['train.parquet']
    frame=pq.read_table(train,columns=['prompt','data_source','extra_info']).to_pandas()
    assert len(frame)==169028
    for _,row in frame.iterrows():
        question,qid=original_question_and_id(row)
        if qid in warm_ids:excluded_contents.add(question_identity(question))
    cache=TokenFeatureCache(original['question_probe']['cache_dir'])
    assert len(cache.ids)==169028 and cache.spec['feature_layer']==27 and cache.spec['global_max_tokens']==244
    selected=[];seen=set();pool=QuestionPool(512)
    for index in np.random.default_rng(args.seed).permutation(len(frame)):
        index=int(index);question,qid=original_question_and_id(frame.iloc[index])
        qhash=question_text_sha(question)
        if qid in exclusions or qhash in excluded_hashes or question_identity(question) in excluded_contents:continue
        if pool.reserve(qid,question) is None:continue
        assert cache.index[qid]['question_hash']==qhash
        selected.append(dict(source_index=index,qid=qid,question=question,question_hash=qhash))
        seen.add(qid)
        if len(selected)==512:break
    assert len(selected)==512 and not seen.intersection(exclusions)
    assert len({question_identity(r['question']) for r in selected})==512
    selection=dict(role='random_new512_canonical_unique_without_warmup_or_seen_formal_candidates',
        seed=args.seed,sampling='numpy.default_rng(seed).permutation(canonical169028), first512 eligible unique',
        canonical_rows=169028,questions=selected,warmup_excluded_count=5120,
        all_excluded_unique_count=len(exclusions),normalized_question_content_deduplication=True,
        formal_candidate_exclusion_receipts=receipts,
        original_train_sha256=sha(train),original_teacher_identity_sha256=sha(remote(TEACHER)/'identity.json'),
        original_head_sha256=HEAD_SHA,feature_cache_dir=str(cache.path),feature_spec_sha256=cache.spec_sha,
        feature_cache_manifest_sha256=sha(cache.path/'manifest.json'),checked_unix=time.time())
    once(out/'selection.json',selection)
    for name,expected in original['source_sha256'].items():
        path=repo/name;assert path.resolve()==path and sha(path)==expected, 'Pinned runtime source differs: '+name
    for name,expected in original['initial_model_sha256'].items():
        assert sha(Path(original['initial_model'])/name)==expected, 'Original official teacher asset differs: '+name
    for name in ('logs','data','teacher-cache','teacher-boundary','artifacts','checkpoints'):(out/name).mkdir(exist_ok=True)
    # Small canonical source files are copied on the remote host. Large model,
    # packed token bank, original rollouts and environment remain read-only.
    import shutil
    for name,expected in original['input_sha256'].items():
        origin=source/'data'/name;target=out/'data'/name
        assert not target.exists();shutil.copyfile(origin,target);assert sha(target)==expected
    run=copy.deepcopy(original)
    run.update(run_id=uuid.uuid4().hex,execution_id=uuid.uuid4().hex,execution_host=socket.gethostname(),
        started_unix=time.time(),completed_updates=0,diagnostic_only=True,
        diagnostic_kind='random_new512_frozen_official_G5_original_head200',
        actor_training_budget=0,head_training_budget=0,source_formal_base=str(source),
        source_formal_run_id=original['run_id'],source_formal_execution_id=original['execution_id'],
        diagnostic_selection_sha256=sha(out/'selection.json'),
        checkpoint_dir=str(out/'checkpoints/search-g1-common8'),train_log=str(out/'logs/diagnostic-teacher.log'),
        gpu_placement_bound=False,training_physical_indices=[],training_gpu_uuids=[],excluded_physical_indices=[])
    for name in ('pipeline_identity','full_state_restore','resume_path','actor_checkpoint','reference_checkpoint',
                 'gpu_placement_binding_receipt','gpu_placement_binding_sha256','gpu_placement_bound_unix',
                 'assigned_gpu_uuids','physical_gpu_inventory','gpu_placement_semantics','ray_tmp_dir_identity'):
        run.pop(name,None)
    run['question_probe']=copy.deepcopy(original['question_probe'])
    run['question_probe'].update(teacher_warmup_cache=str(out/'teacher-cache'),output_dir=str(out/'artifacts/question-probe-unused'))
    run['ray_tmp_dir']='/tmp/g-'+run['execution_id']
    once(out/'logs/run-manifest.json',run)
    document=copy.deepcopy(read(source/'logs/resolved-training-config.json')['resolved_config'])
    assert document['actor_rollout_ref']['rollout']['async_phase_enabled'] is True
    assert document['actor_rollout_ref']['rollout']['temperature']==1. and document['actor_rollout_ref']['rollout']['top_p']==1.
    assert document['actor_rollout_ref']['model']['path']==run['initial_model']
    assert not document['trainer'].get('full_state_restore_receipt') and not document['actor_rollout_ref']['model'].get('actor_checkpoint_path')
    document['data']['train_files']=str(out/'data/train.parquet');document['data']['val_files']=str(out/'data/stage.parquet')
    document['trainer']['default_local_dir']=run['checkpoint_dir']
    if 'question_probe' in document:document['question_probe']=copy.deepcopy(run['question_probe'])
    once(out/'teacher-config.json',document)
    result=dict(role='new512_diagnostic_inputs_staged',run_id=run['run_id'],execution_id=run['execution_id'],
        selected_unique=512,warmup_overlap=0,observed_formal_candidate_overlap=0,seed=args.seed,
        source_commit=RUNTIME_SHA,selection_sha256=sha(out/'selection.json'),config_sha256=sha(out/'teacher-config.json'),
        actor_training_budget=0,head_training_budget=0,feature_reencoding=False,
        output_dir=str(out),required_training_gpus=[0,1,2,3],required_retriever_gpus=[5,6],
        reward_probe_endpoint_required=False,ray_cpus=16,ray_object_store_bytes=8*1024**3,checked_unix=time.time())
    once(out/'stage-complete.json',result);print(json.dumps(result),flush=True)


def teacher_fit(trainer,settings):
    """The original frozen-teacher G5 generation/scoring segment, with no fit."""
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from verl import DataProto
    from search_r1.llm_agent.generation import LLMGenerationManager,GenerationConfig
    from search_g1.rollout_phase_contract import configure_generation,scalar_metrics
    from search_g1.qprobe_runtime import QProbeCandidateStream,_complete_generation,_contracts
    from search_g1.mixed_runtime import strict_vectors
    from search_g1.qprobe_warmup import moments,verify_frozen,teacher_identity
    from search_g1.training_state import atomic_save
    from reproduction.common8.behavior_runtime import prepare_identity,attach_generation_identity
    base=Path(settings['output_dir']);run=read(base/'logs/run-manifest.json');selection=read(base/'selection.json')
    assert run['diagnostic_only'] and run['actor_training_budget']==run['head_training_budget']==0
    assert len(trainer.train_dataloader.dataset.dataframe)==169028
    rows=selection['questions'];assert len(rows)==512
    trainer.global_steps=1
    config=GenerationConfig(max_turns=trainer.config.max_turns,
        max_start_length=trainer.config.data.max_start_length,max_prompt_length=trainer.config.data.max_prompt_length,
        max_response_length=trainer.config.data.max_response_length,max_obs_length=trainer.config.data.max_obs_length,
        num_gpus=4,no_think_rl=trainer.config.algorithm.no_think_rl,
        search_url=trainer.config.retriever.url,topk=trainer.config.retriever.topk)
    configure_generation(config,trainer.config.actor_rollout_ref.rollout);assert config.async_phase_enabled is True
    manager=LLMGenerationManager(tokenizer=trainer.tokenizer,actor_rollout_wg=trainer.actor_rollout_wg,config=config)
    stream=QProbeCandidateStream(trainer.train_dataloader)
    candidates=DataProto.from_dict(tensors={'qprobe_source_index':torch.tensor([r['source_index'] for r in rows])},
        non_tensors={'candidate_original_question':np.array([r['question'] for r in rows],dtype=object),
                     'candidate_stable_id':np.array([r['qid'] for r in rows],dtype=object)})
    parent=stream.materialize(candidates,list(range(512)))
    bound=teacher_identity(run,feature_spec_sha=selection['feature_spec_sha256'],
        sampling_config=OmegaConf.to_container(trainer.config.actor_rollout_ref.rollout,resolve=True),
        agent_config={k:getattr(trainer.config.data,k) for k in ('max_start_length','max_prompt_length','max_response_length','max_obs_length')},
        max_turns=trainer.config.max_turns,seed=selection['seed'],
        questions=[dict(qid=r['qid'],question_hash=r['question_hash']) for r in rows])
    once(base/'teacher-cache/identity.json',bound)
    print(json.dumps(dict(stage='NEW512_FRESH_OFFICIAL_TEACHER_BOUNDARY_BEFORE',questions=512,trajectories=2560)),flush=True)
    before=trainer.actor_rollout_wg.capture_frozen_teacher_boundary(str(base/'teacher-boundary/before'))
    identity=prepare_identity(parent,0,repeats=5,teacher_warmup=True,candidate_offset=0)
    repeated=parent.repeat(repeat_times=5,interleave=True)
    gen=repeated.pop(batch_keys=['input_ids','attention_mask','position_ids']);attach_generation_identity(gen,identity,do_sample=True)
    manager.timing_raw={}
    output=manager.run_llm_loop(gen_batch=gen,initial_input_ids=gen.batch['input_ids'][:,-config.max_start_length:].clone().long())
    _complete_generation(output,identity)
    contracts=_contracts(trainer,output,run)
    vectors=strict_vectors(parent,output,trainer.tokenizer);assert len(vectors)==512
    print(json.dumps(dict(stage='NEW512_REAL_G5_GENERATION_AND_STRICT_EM_COMPLETE',questions=512,trajectories=2560)),flush=True)
    after=trainer.actor_rollout_wg.capture_frozen_teacher_boundary(str(base/'teacher-boundary/after'))
    frozen=verify_frozen(before,after,4)
    restored=trainer.actor_rollout_wg.restore_frozen_teacher_rng(str(base/'teacher-boundary/before'))
    assert len(restored)==4 and all(r['rng_equal'] for r in restored)
    raw_path=base/'teacher-cache/rollout.pt'
    atomic_save(raw_path,dict(output=output,identity=identity,contracts=contracts,frozen_workers=frozen,
        question_bindings=bound['questions']))
    groups=[]
    for i,(row,vector) in enumerate(zip(rows,vectors)):
        count,mean,std=moments(vector)
        groups.append(dict(qid=row['qid'],question_hash=row['question_hash'],strict_em=vector,
            correct_count=count,mean=mean,std=std,trajectory_ids=identity['trajectory_ids'][i*5:(i+1)*5]))
    assert len({t for g in groups for t in g['trajectory_ids']})==2560
    labels=dict(role='actual_frozen_official_new512_G5_labels',questions=512,trajectories=2560,groups=groups,
        run_id=run['run_id'],execution_id=run['execution_id'],selection_sha256=sha(base/'selection.json'),
        teacher_identity_sha256=sha(base/'teacher-cache/identity.json'),raw_rollout_sha256=sha(raw_path),
        raw_rollout_bytes=raw_path.stat().st_size,frozen_workers=frozen,actor_rng_restored=restored,
        contracts=contracts,actual_async_rollout=scalar_metrics(output),timing_raw=manager.timing_raw,
        actual_actor_optimizer_updates=0,actual_head_optimizer_updates=0,checked_unix=time.time())
    once(base/'teacher-cache/labels.json',labels)
    summary={k:v for k,v in labels.items() if k not in ('groups','frozen_workers','actor_rng_restored','contracts')}
    summary.update(stage='NEW512_FROZEN_TEACHER_LABELS_COMPLETE',
        true_mixed_fraction=sum(0<g['correct_count']<5 for g in groups)/512,
        correct_count_histogram={str(k):sum(g['correct_count']==k for g in groups) for k in range(6)})
    once(base/'teacher-complete.json',summary);print(json.dumps(summary),flush=True)


def teacher_task(document,settings):
    from omegaconf import OmegaConf
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    from verl.trainer.main_ppo_eco import run_main_task
    def diagnostic_fit(self):return teacher_fit(self,settings)
    RayPPOTrainer.fit=diagnostic_fit
    # run_main_task retains the actual four-rank fresh FSDP/vLLM workers and
    # original tokenizer/dataloader. Its fit dispatch reaches only the frozen
    # generation segment above; QProbeRuntime and all training loops are absent.
    run_main_task(OmegaConf.create(document))


def teacher_child(args,out):
    """A real fresh driver exec, born with the first admitted training UUID.

    Reuse the published native driver's visibility/RNG checks without inventing
    a formal training shell claim for this diagnostic-only generation.
    """
    import ctypes
    import pickle
    import signal
    from reproduction.a800.owned_processes import identity,same
    parent={k:v for k,v in identity(os.getppid()).items() if k in ('pid','start_ticks','sid','uid')}
    assert ctypes.CDLL(None,use_errno=True).prctl(1,signal.SIGTERM,0,0,0)==0
    assert same(parent), 'Diagnostic supervisor died during child bootstrap'
    payload_path=out/'diagnostic-driver-payload.pkl'
    claim=read(out/'diagnostic-driver-once.json');assert sha(payload_path)==claim['payload_sha256']
    payload=pickle.loads(payload_path.read_bytes())
    assert payload['parent_identity']==parent and claim['parent_identity']==parent
    run=read(out/'logs/run-manifest.json');allocation=read(out/'logs/native-allocation-admission.json')
    assert run['diagnostic_only'] and run['actor_training_budget']==run['head_training_budget']==0
    assert claim['run_id']==run['run_id'] and claim['execution_id']==run['execution_id']
    assert claim['allocation_sha256']==sha(out/'logs/native-allocation-admission.json')
    assert claim['placement_sha256']==sha(out/run['gpu_placement_binding_receipt'])==run['gpu_placement_binding_sha256']
    route=payload['route'];assert route==claim['route']
    assert route['gpu_uuid']==run['training_gpu_uuids'][0]==allocation['actual_cuda_runtime_uuids'][0]
    assert route['physical_index']==run['training_physical_indices'][0]
    initial=dict(item.split(b'=',1) for item in Path('/proc/self/environ').read_bytes().split(b'\0') if b'=' in item)
    born=initial.get(b'CUDA_VISIBLE_DEVICES',b'').decode()
    assert born==os.environ.get('CUDA_VISIBLE_DEVICES')==route['gpu_uuid']
    import ray
    from search_g1.qprobe_driver_gpu import (RAY_RUNTIME_ENV,verify_driver_visibility,
        restore_cpu_rng,cpu_rng_sha256,_cluster_identity)
    assert not ray.is_initialized()
    cluster=payload['cluster'];assert _cluster_identity(cluster['ray_gcs_address'],cluster['ray_namespace'])==cluster
    ray.init(address=cluster['ray_gcs_address'],namespace=cluster['ray_namespace'],runtime_env=RAY_RUNTIME_ENV)
    try:
        context=ray.get_runtime_context()
        assert _cluster_identity(context.gcs_address,context.namespace)==cluster and ray.get_gpu_ids()==[]
        import torch
        from reproduction.common8.native_allocation import cuda_uuids
        actual=verify_driver_visibility(route,environ=os.environ,birth_visible=born,
            cuda_initialized=torch.cuda.is_initialized,cuda_device_count=torch.cuda.device_count,cuda_uuids=cuda_uuids)
        # Match the published native driver: load the entry module before
        # restoring transferred CPU RNG, so imports cannot perturb that state.
        from verl.trainer.main_ppo_eco import run_main_task as native_entry
        assert restore_cpu_rng(payload['cpu_rng'])==payload['cpu_rng_sha256']==cpu_rng_sha256(payload['cpu_rng'])
        once(out/'diagnostic-driver-route.json',dict(role='actual_fresh_exec_diagnostic_driver',
            run_id=run['run_id'],execution_id=run['execution_id'],route=route,cluster=cluster,**actual,
            producer_identity=identity(os.getpid()),parent_identity=parent,
            formal_training_claim_created=False,original_native_visibility_guard_reused=True,checked_unix=time.time()))
        teacher_task(read(out/'teacher-config.json'),dict(output_dir=str(out)))
    finally:ray.shutdown()
    once(out/'diagnostic-driver-exit.json',dict(exit_code=0,run_id=run['run_id'],execution_id=run['execution_id'],checked_unix=time.time()))


def teacher(args,out):
    import ray
    assert not ray.is_initialized() and os.environ.get('SLURM_JOB_ID') is None
    run=read(out/'logs/run-manifest.json');staged=read(out/'stage-complete.json')
    assert run['diagnostic_only'] and run['diagnostic_selection_sha256']==sha(out/'selection.json')==staged['selection_sha256']
    assert staged['config_sha256']==sha(out/'teacher-config.json')
    os.environ.update(SEARCH_G1_BASE=str(out),SEARCH_G1_PROFILE='common8',SEARCH_G1_TRAINING_WORLD_SIZE='4',
        CUDA_VISIBLE_DEVICES='0,1,2,3',PYTHONPATH=str(args.runtime_repo),NCCL_NVLS_ENABLE='0',VLLM_ATTENTION_BACKEND='XFORMERS',
        TOKENIZERS_PARALLELISM='true',OMP_NUM_THREADS='8',PYTHONUNBUFFERED='1',NO_PROXY='127.0.0.1,localhost',
        C_INCLUDE_PATH='/data/search-r1/bootstrap-tools/python-dev/usr/include/python3.10:/data/search-r1/bootstrap-tools/python-dev/usr/include')
    for key in ('RAY_ADDRESS','PYTHONHOME','LD_LIBRARY_PATH','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):
        os.environ.pop(key,None)
    from reproduction.a800.owned_processes import identity
    from reproduction.common8.native_allocation import bind_idle_local_placement,admit
    run=bind_idle_local_placement(out)
    assert run['training_physical_indices']==[0,1,2,3], 'Diagnostic assignment may not expand beyond authorized training GPUs0..3'
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(map(str,run['training_physical_indices']))
    once(out/'teacher-generation-once.json',dict(run_id=run['run_id'],execution_id=run['execution_id'],
        supervisor=identity(os.getpid()),actor_training_forbidden=True,head_training_forbidden=True,claimed_unix=time.time()))
    once(out/'teacher-launch.json',dict(run_id=run['run_id'],execution_id=run['execution_id'],
        supervisor=identity(os.getpid()),argv=sys.argv,source_commit=RUNTIME_SHA,
        operator_sha256=sha(__file__),runtime_repo=str(args.runtime_repo),
        stdout_stderr_policy='inherited caller streams; caller must retain launch logs',
        tmpdir=os.environ['TMPDIR'],ray_tmpdir=run['ray_tmp_dir'],launched_unix=time.time()))
    locks=[];root=Path('/data/shaoshitong/Search-G1-qprobe-shared/token-feature-gpu-locks');root.mkdir(exist_ok=True,parents=True)
    for gpu_uuid in run['assigned_gpu_uuids'].values():
        lock=(root/(gpu_uuid+'.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(lock)
    admission=admit(out)
    assert admission['pipeline_identity']['pid']==os.getpid() and admission['node']==socket.gethostname()
    assert admission['actual_allocated_indices']==list(ALLOWED)
    from reproduction.a800.owned_processes import publish
    run['pipeline_identity']=admission['pipeline_identity'];publish(out/'logs/run-manifest.json',run)
    # The unchanged local corpus extractor asks for the remaining checkpoint
    # allowance. This diagnostic's actual checkpoint budget is explicitly zero.
    # It must still reserve the original corpus JSONL/Arrow cache disk floor.
    import shutil
    once(out/'logs/checkpoint-disk-admission.json',dict(accepted=True,run_id=run['run_id'],
        role='diagnostic_zero_training_checkpoint_budget',save_steps=[],required_free_bytes=0,
        actual_free_bytes=shutil.disk_usage(out).free,diagnostic_only=True,checked_unix=time.time()))
    from reproduction.common8.run_pipeline import Pipeline
    pipeline=Pipeline(out,run)
    try:
        pipeline.stage('DIAGNOSTIC_STAGING_ORIGINAL_LOCAL_RETRIEVAL')
        stager=pipeline.start('diagnostic-retrieval-stage',[sys.executable,'-m','reproduction.common8.stage_retrieval','--base',str(out)],[])
        pipeline.wait('diagnostic-retrieval-stage',stager)
        import importlib.util
        nvidia=importlib.util.find_spec('nvidia');torch_spec=importlib.util.find_spec('torch')
        assert nvidia is not None and nvidia.submodule_search_locations and torch_spec is not None and torch_spec.origin
        libs=list(dict.fromkeys(p for root in nvidia.submodule_search_locations
            for p in sorted(Path(root).glob('*/lib')) if p.is_dir()))+[Path(torch_spec.origin).parent/'lib']
        pipeline.stage('DIAGNOSTIC_STARTING_OWNED_ORIGINAL_TWO_SHARD_RETRIEVER')
        retriever=pipeline.start('diagnostic-retriever',[sys.executable,'-m','reproduction.common8.retriever'],[5,6],
            {'PYTHONPATH':run['retriever_gpu_lib']+':'+str(args.runtime_repo),'LD_LIBRARY_PATH':':'.join(map(str,libs))})
        health=pipeline.endpoint('diagnostic-retriever',retriever,'18010/health')
        assert health['ok'] is True and health['run_id']==run['run_id'] and health['execution_id']==run['execution_id']
        assert health['shard_count']==2 and health['slurm_job_id'] is None
        assert health['admission_sha256']==sha(out/'logs/retrieval-admission.json')
        once(out/'diagnostic-retriever-ready.json',dict(health=health,owned_leader=pipeline.owned.jobs['diagnostic-retriever']['leader'],
            native_allocation_sha256=sha(out/'logs/native-allocation-admission.json'),checked_unix=time.time()))
        pipeline.stage('DIAGNOSTIC_PRIVATE_RAY_FRESH_OFFICIAL_FROZEN_TEACHER')
        ray_root=Path(run['ray_tmp_dir']);ray_root.mkdir(mode=0o700,exist_ok=False)
        try:
            ray.init(address='local',num_cpus=16,num_gpus=4,object_store_memory=8*1024**3,_temp_dir=str(ray_root),
                runtime_env={'env_vars':{'TOKENIZERS_PARALLELISM':'true','NCCL_DEBUG':'WARN'}})
            import pickle
            from search_g1.qprobe_driver_gpu import capture_cpu_rng,cpu_rng_sha256,child_environment,_cluster_identity
            context=ray.get_runtime_context();cluster=_cluster_identity(context.gcs_address,context.namespace)
            state=capture_cpu_rng();route=dict(gpu_uuid=run['training_gpu_uuids'][0],physical_index=run['training_physical_indices'][0])
            parent_birth={k:v for k,v in identity(os.getpid()).items() if k in ('pid','start_ticks','sid','uid')}
            payload=dict(route=route,cluster=cluster,cpu_rng=state,cpu_rng_sha256=cpu_rng_sha256(state),parent_identity=parent_birth)
            raw=pickle.dumps(payload,protocol=4);assert len(raw)<=4*1024**2
            with (out/'diagnostic-driver-payload.pkl').open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
            once(out/'diagnostic-driver-once.json',dict(run_id=run['run_id'],execution_id=run['execution_id'],
                parent_identity=payload['parent_identity'],route=route,payload_sha256=sha(out/'diagnostic-driver-payload.pkl'),
                allocation_sha256=sha(out/'logs/native-allocation-admission.json'),
                placement_sha256=run['gpu_placement_binding_sha256'],formal_training_claim_created=False,claimed_unix=time.time()))
            env=child_environment(route,pipeline.env)
            child=pipeline.owned.start('diagnostic-teacher-driver',[sys.executable,str(Path(__file__).resolve()),
                '--action','teacher-child','--runtime-repo',str(args.runtime_repo),'--formal-base',str(args.formal_base),
                '--output-dir',str(out)],env,out/'logs/diagnostic-teacher-driver.log',Path(args.runtime_repo))
            print(json.dumps(dict(stage='DIAGNOSTIC_FRESH_BIRTH_UUID_DRIVER_STARTED',pid=child.pid,
                gpu_uuid=route['gpu_uuid'],log=str(out/'logs/diagnostic-teacher-driver.log'))),flush=True)
            pipeline.wait('diagnostic-teacher-driver',child)
        finally:ray.shutdown()
    finally:
        pipeline.owned.stop_all()
        if (out/'logs/corpus-cache-claim.json').exists():
            from reproduction.a800_common.cache import cleanup
            cleanup(out)
    assert (out/'teacher-complete.json').exists()
    once(out/'teacher-exit.json',dict(exit_code=0,run_id=run['run_id'],execution_id=run['execution_id'],
        actual_actor_optimizer_updates=0,actual_head_optimizer_updates=0,checked_unix=time.time()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--action',required=True,choices=('compare','replay-control','infer','stage','teacher','teacher-child'))
    p.add_argument('--runtime-repo',required=True);p.add_argument('--formal-base',required=True)
    p.add_argument('--output-dir',required=True);p.add_argument('--warmup-dir',default=WARMUP)
    p.add_argument('--physical-index',type=int);p.add_argument('--labels')
    p.add_argument('--seed',type=int,default=2026101601)
    args=p.parse_args();runtime,out=bind(args)
    if args.action=='stage':
        os.environ['CUDA_VISIBLE_DEVICES']='';stage(args,out);return
    if args.action=='teacher-child':
        try:teacher_child(args,out)
        except BaseException as error:
            once(out/'diagnostic-driver-failure.json',dict(error=repr(error),traceback=traceback.format_exc(),
                pid=os.getpid(),automatic_retry=False,failed_unix=time.time()))
            raise
        return
    if args.action=='teacher':
        try:teacher(args,out)
        except BaseException as error:
            failure=dict(role='new512_frozen_teacher_failure',error=repr(error),traceback=traceback.format_exc(),
                pid=os.getpid(),source_commit=RUNTIME_SHA,automatic_retry=False,failed_unix=time.time())
            once(out/'teacher-failure.json',failure)
            if not (out/'teacher-exit.json').exists():once(out/'teacher-exit.json',dict(exit_code=1,**failure))
            raise
        return
    if args.action=='compare':
        os.environ['CUDA_VISIBLE_DEVICES']=''
        ready,raw,head,formal,identity=heads(args);result=compare_heads(head,formal,identity)
        once(out/'tensor-comparison.json',result);print(json.dumps({k:v for k,v in result.items() if k!='tensors'}),flush=True);return
    assert args.physical_index in ALLOWED
    os.environ['CUDA_VISIBLE_DEVICES']=str(args.physical_index)
    from reproduction.a800_common.qprobe_token_warmup import gpu_admission
    lock,proof=gpu_admission(args.physical_index);once(out/'gpu-admission.json',proof)
    import torch
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    if args.action=='replay-control':replay_control(args,out)
    else:
        assert args.labels;infer(args,out)


if __name__=='__main__':main()
