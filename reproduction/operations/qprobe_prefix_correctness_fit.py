"""Offline prefix5 trajectory-correctness fit; never starts or updates an actor.

Run as a standalone file in a fresh process, after the feature producer exits.
Large features/checkpoints remain on remote storage. The fixed test split is
reported at epoch0 and every epoch; epoch20 is always the final model.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode=True
RUNTIME_COMMIT='fcb77204f48d5cb8ec21c68df7d4a2c937b3557a'
REVISION='aa8e72537993ba99e69dfaafa59ed015b17504d1'
REPRESENTATION='actual_prompt_policy_prefix5_alltokens_layer27_v1'
ROLE='qwen_native_prefix5_trajectory_correctness_warmup20'
FIT=dict(epochs=20,batch_size=256,micro_batch_size=32,learning_rate=.0003,seed=1729,
    loss='BCEWithLogits',weight_decay=0.,class_weights=None,outputs=1,
    training_questions=5120,training_trajectories=25600,test_questions=512,test_trajectories=2560,
    expected_head_adam_updates=2000,final_epoch=20,test_used_for_optimization=False)


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()


def remote(value):
    path=Path(value)
    if not path.is_absolute() or path.resolve()!=path or not str(path).startswith(('/data/','/mnt/','/backup/')):
        raise ValueError('PREFIX_FIT_REQUIRES_CANONICAL_REMOTE_STORAGE')
    return path


def once(path,value):
    raw=canonical(value)+b'\n';fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())


def append(path,value):
    with path.open('ab') as stream:stream.write(canonical(value)+b'\n');stream.flush();os.fsync(stream.fileno())


def event(output,stage,**values):
    row=dict(stage=stage,observed_unix=time.time(),**values)
    append(output/'progress.jsonl',row)
    print(json.dumps(row,allow_nan=False),flush=True)


def born_gpu_uuid(index):
    entries=Path('/proc/self/environ').read_bytes().split(b'\0')
    values=[item.split(b'=',1)[1].decode('ascii') for item in entries if item.startswith(b'CUDA_VISIBLE_DEVICES=')]
    if len(values)!=1 or not re.fullmatch(r'GPU-[0-9a-fA-F-]+',values[0]) or os.environ.get('CUDA_VISIBLE_DEVICES')!=values[0]:
        raise ValueError('PREFIX_FIT_REQUIRES_SINGLE_GPU_UUID_AT_PROCESS_BIRTH')
    actual=subprocess.check_output(['nvidia-smi','--id='+str(index),'--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).strip()
    if actual!=values[0]:raise ValueError('PREFIX_FIT_BORN_UUID_DOES_NOT_MATCH_ACTUAL_PHYSICAL_GPU')
    return values[0]


def validate_cache(cache):
    spec=cache.spec
    if (spec.get('representation')!=REPRESENTATION or spec.get('encoder_revision')!=REVISION or
            spec.get('feature_layer')!=27 or spec.get('feature_dim')!=2048 or
            spec.get('encoder_compute_dtype')!='bfloat16' or spec.get('feature_dtype')!='bfloat16' or
            spec.get('split_question_isolation') is not True or type(cache.max_tokens) is not int or
            not 0<cache.max_tokens<=32768 or spec.get('global_max_tokens')!=cache.max_tokens):
        raise ValueError('PREFIX_FIT_CACHE_REPRESENTATION_OR_ENCODER_CHANGED')
    ids={key:list(cache.ids_by_split[key]) for key in ('train','test')}
    if (len(ids['train'])!=25600 or len(ids['test'])!=2560 or
            len(set(ids['train']+ids['test']))!=28160 or set(cache.ids)!=set(ids['train']+ids['test'])):
        raise ValueError('PREFIX_FIT_REQUIRES_EXACT_25600_TRAIN_AND_2560_FIXED_TEST')
    questions,hashes,positive={}, {}, {}
    for split,size in (('train',5120),('test',512)):
        rows=[cache.index[tid] for tid in ids[split]]
        if any(row['split']!=split or type(row['strict_em']) is not int or row['strict_em'] not in (0,1) for row in rows):
            raise ValueError('PREFIX_FIT_REQUIRES_ACTUAL_TRAJECTORY_BINARY_LABELS')
        counts=Counter(row['qid'] for row in rows)
        if len(counts)!=size or set(counts.values())!={5}:
            raise ValueError('PREFIX_FIT_REQUIRES_COMPLETE_G5_WITHIN_EACH_QUESTION_SPLIT')
        by_qid={}
        for row in rows:
            if type(row['length']) is not int or not 0<row['length']<=cache.max_tokens:
                raise ValueError('PREFIX_FIT_TOKEN_LENGTH_CHANGED')
            if row['qid'] in by_qid and by_qid[row['qid']]!=row['question_hash']:
                raise ValueError('PREFIX_FIT_QUESTION_HASH_CHANGED_WITHIN_G5')
            by_qid[row['qid']]=row['question_hash']
        questions[split]=set(by_qid);hashes[split]=set(by_qid.values())
        positive[split]=sum(row['strict_em'] for row in rows)
    if questions['train']&questions['test'] or hashes['train']&hashes['test']:
        raise ValueError('PREFIX_FIT_FIXED_TEST_IS_NOT_QUESTION_DISJOINT')
    if max(cache.index[tid]['length'] for tid in cache.ids)!=cache.max_tokens:
        raise ValueError('PREFIX_FIT_BMAX_MUST_MATCH_ALL_CACHE_INPUTS')
    proof=cache.manifest['split_proof']
    if proof.get('disjoint_qid') is not True or proof.get('disjoint_question_hash') is not True:
        raise ValueError('PREFIX_FIT_SPLIT_PRODUCER_PROOF_MISSING')
    return ids,positive


def prepare_batch(features,mask,positions,*,hidden_size,max_tokens,max_positions):
    import torch
    if (features.ndim!=3 or features.shape[-1]!=hidden_size or not len(features) or
            features.shape[1]>max_tokens or mask.shape!=features.shape[:2] or positions.shape!=mask.shape or
            mask.dtype!=torch.bool or positions.dtype!=torch.long):
        raise ValueError('PREFIX_FIT_BATCH_SHAPE_OR_MASK_CHANGED')
    lengths=mask.sum(-1)
    if bool((lengths<1).any()):raise ValueError('PREFIX_FIT_EMPTY_PREFIX')
    local=int(lengths.max());expected=torch.arange(mask.shape[1],device=mask.device)[None,:]<lengths[:,None]
    if not torch.equal(mask,expected):raise ValueError('PREFIX_FIT_REQUIRES_RIGHT_PADDING_WITHOUT_TOKEN_REMOVAL')
    features=features[:,:local].detach();mask=mask[:,:local];positions=positions[:,:local]
    if bool(((positions<0)|(positions>=max_positions))[mask].any()):raise ValueError('PREFIX_FIT_ACTUAL_POSITION_OUT_OF_NATIVE_RANGE')
    if local>1 and bool(((positions[:,1:]<=positions[:,:-1])&mask[:,1:]).any()):
        raise ValueError('PREFIX_FIT_ACTUAL_POSITIONS_NOT_CAUSAL')
    features=features.masked_fill(~mask[:,:,None],0.)
    if not torch.isfinite(features).all():raise ValueError('PREFIX_FIT_NONFINITE_REAL_TOKEN_FEATURE')
    return features,mask,positions.masked_fill(~mask,0)


def build_probe(config):
    """Reuse native Qwen and the exact-copy method; instantiate only Linear1."""
    import torch
    from torch import nn
    from search_g1.qprobe_qwen_block_model import QwenLastBlockQuestionProbe
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer,Qwen2RotaryEmbedding
    from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask_for_sdpa
    class PrefixCorrectnessProbe(QwenLastBlockQuestionProbe):
        def __init__(self):
            nn.Module.__init__(self);self.config=config;native=config.native_config()
            self.block=Qwen2DecoderLayer(native,layer_idx=config.source_layer)
            self.rotary=Qwen2RotaryEmbedding(config=native)
            self.output=nn.Linear(config.max_tokens*config.hidden_size,1)
            self._forward_parameters=inspect.signature(self.block.forward).parameters
            if 'position_embeddings' not in self._forward_parameters:raise ValueError('PREFIX_FIT_NATIVE_ROTARY_API_CHANGED')

        def forward(self,features,mask,positions):
            features,mask,positions=prepare_batch(features,mask,positions,hidden_size=config.hidden_size,
                max_tokens=config.max_tokens,max_positions=config.max_position_embeddings)
            features=features.to(dtype=self.output.weight.dtype)
            causal=_prepare_4d_causal_attention_mask_for_sdpa(mask,features.shape[:2],features,0)
            kwargs=dict(attention_mask=causal,position_ids=positions,position_embeddings=self.rotary(features,positions),
                use_cache=False,output_attentions=False)
            if 'cache_position' in self._forward_parameters:kwargs['cache_position']=torch.arange(features.shape[1],device=features.device)
            hidden=self.block(features,**kwargs)
            if isinstance(hidden,tuple):hidden=hidden[0]
            if hidden.shape!=features.shape:raise ValueError('PREFIX_FIT_NATIVE_OUTPUT_SHAPE_CHANGED')
            hidden=hidden.masked_fill(~mask[:,:,None],0.)
            hidden=torch.nn.functional.pad(hidden,(0,0,0,config.max_tokens-hidden.shape[1]))
            logits=self.output(hidden.flatten(start_dim=1)).squeeze(-1)
            if not torch.isfinite(logits).all():raise ValueError('PREFIX_FIT_NONFINITE_CORRECTNESS_LOGIT')
            return logits
    return PrefixCorrectnessProbe()


def binary_metrics(logits,labels,*,training_prior):
    import torch
    if logits.ndim!=1 or labels.shape!=logits.shape or not len(logits) or not torch.isfinite(logits).all() or not torch.isin(labels,torch.tensor([0.,1.])).all():
        raise ValueError('PREFIX_FIT_INVALID_BINARY_METRIC_INPUT')
    truth=labels.bool();prediction=logits>=0;size=len(labels)
    tp,tn,fp,fn=[int(value.sum()) for value in (truth&prediction,~truth&~prediction,~truth&prediction,truth&~prediction)]
    nll=float(torch.nn.functional.binary_cross_entropy_with_logits(logits.double(),labels.double()))
    positive=tp+fn;majority=max(positive,size-positive)/size
    prior=min(max(training_prior,1e-12),1-1e-12)
    prior_nll=-(positive*math.log(prior)+(size-positive)*math.log1p(-prior))/size
    return dict(samples=size,positive_count=positive,negative_count=size-positive,positive_fraction=positive/size,
        binary_bce=nll,nll=nll,binary_accuracy=(tp+tn)/size,threshold=.5,threshold_tie_count=int((logits==0).sum()),
        confusion=dict(tp=tp,tn=tn,fp=fp,fn=fn),predicted_positive_count=tp+fp,predicted_positive_fraction=(tp+fp)/size,
        always_correct_baseline_accuracy=positive/size,always_incorrect_baseline_accuracy=(size-positive)/size,
        majority_baseline_accuracy=majority,accuracy_above_majority_baseline=(tp+tn)/size-majority,
        constant_training_prior_probability=training_prior,constant_training_prior_nll=prior_nll)


def adam_steps(optimizer):
    values=[int(state['step'].item()) for state in optimizer.state.values() if 'step' in state]
    if values and len(set(values))!=1:raise ValueError('PREFIX_FIT_ADAM_COUNTER_DISAGREEMENT')
    return values[0] if values else 0


def evaluate(model,cache,ids,*,epoch,split,updates,training_prior):
    import torch
    prior=model.training;logits=[];labels=[]
    try:
        model.eval()
        with torch.no_grad():
            for start in range(0,len(ids),FIT['micro_batch_size']):
                selected=ids[start:start+FIT['micro_batch_size']];batch=cache.get_batch(selected,device='cuda:0')
                value=model(batch['features'],batch['attention_mask'],batch['position_ids'])
                logits.append(value.detach().cpu());labels.extend(cache.index[tid]['strict_em'] for tid in selected)
    finally:model.train(prior)
    result=binary_metrics(torch.cat(logits),torch.tensor(labels,dtype=torch.float32),training_prior=training_prior)
    return dict(result,epoch=epoch,split=split,questions=len({cache.index[tid]['qid'] for tid in ids}),
        actual_head_adam_updates=updates,scope='all_training_trajectories' if split=='train' else 'fixed_question_disjoint_test',
        used_for_optimization=split=='train')


def checkpoint(path,model,optimizer,rng,*,epoch,updates,binding):
    import torch
    temporary=path.with_suffix('.partial.pt')
    if path.exists() or temporary.exists():raise FileExistsError(path)
    state=dict(schema_version=1,role=ROLE,epoch=epoch,actual_head_adam_updates=updates,
        config=FIT,model=model.state_dict(),optimizer=optimizer.state_dict(),private_rng=rng.get_state(),binding=binding,
        actor_updated=False,formal_training_started=False)
    with temporary.open('xb') as stream:torch.save(state,stream);stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)
    return dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size,epoch=epoch,actual_head_adam_updates=updates)


def fit(model,cache,ids,output,binding):
    import torch
    optimizer=torch.optim.Adam(model.parameters(),lr=FIT['learning_rate'],weight_decay=0.)
    rng=torch.Generator(device='cpu').manual_seed(FIT['seed']+1)
    epochs=[];minis=[];saved=[];updates=0;prior=model.training
    training_prior=sum(cache.index[tid]['strict_em'] for tid in ids['train'])/len(ids['train'])
    try:
        for epoch in range(FIT['epochs']+1):
            if epoch:
                model.train();order=torch.randperm(len(ids['train']),generator=rng)
                for batch_index,index in enumerate(order.split(FIT['batch_size'])):
                    optimizer.zero_grad(set_to_none=True);loss_sum=0.;correct=0;before=adam_steps(optimizer)
                    for micro in index.split(FIT['micro_batch_size']):
                        selected=[ids['train'][i] for i in micro.tolist()];batch=cache.get_batch(selected,device='cuda:0')
                        truth=torch.tensor([cache.index[tid]['strict_em'] for tid in selected],device='cuda:0',dtype=torch.float32)
                        logits=model(batch['features'],batch['attention_mask'],batch['position_ids'])
                        loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,truth)
                        fraction=len(micro)/len(index);(loss*fraction).backward();loss_sum+=float(loss.detach())*fraction
                        correct+=int(((logits.detach()>=0)==truth.bool()).sum())
                    if not math.isfinite(loss_sum) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                        raise ValueError('PREFIX_FIT_NONFINITE_PRESTEP_LOSS_OR_GRADIENT')
                    optimizer.step();updates+=1
                    row=dict(epoch=epoch,minibatch_index=batch_index,batch_size=len(index),micro_batch_size=FIT['micro_batch_size'],
                        learning_rate=float(optimizer.param_groups[0]['lr']),head_update_ordinal=updates,
                        head_adam_steps_before=before,head_adam_steps_after=adam_steps(optimizer),
                        pre_step_binary_bce=loss_sum,pre_step_nll=loss_sum,pre_step_binary_accuracy=correct/len(index))
                    minis.append(row);append(output/'minibatch-metrics.jsonl',row)
                    if row['head_adam_steps_after']!=updates or any(not torch.isfinite(p).all() for p in model.parameters()):
                        raise ValueError('PREFIX_FIT_NONFINITE_PARAMETERS_OR_ACTUAL_ADAM_COUNT')
            state=rng.get_state().clone();before=adam_steps(optimizer)
            row=dict(epoch=epoch,actual_head_adam_updates=updates,
                train=evaluate(model,cache,ids['train'],epoch=epoch,split='train',updates=updates,training_prior=training_prior),
                test=evaluate(model,cache,ids['test'],epoch=epoch,split='test',updates=updates,training_prior=training_prior))
            if not torch.equal(state,rng.get_state()) or adam_steps(optimizer)!=before:raise ValueError('PREFIX_FIT_EVALUATION_MUTATED_TRAINING_STATE')
            receipt=checkpoint(output/('head-epoch-%02d.pt'%epoch),model,optimizer,rng,epoch=epoch,updates=updates,binding=binding)
            row['checkpoint']=receipt;saved.append(receipt);epochs.append(row)
            once(output/('epoch-%02d.json'%epoch),row);event(output,'epoch_train_and_fixed_test_complete',**row)
        if updates!=2000 or adam_steps(optimizer)!=2000 or len(minis)!=2000 or len(epochs)!=21:
            raise ValueError('PREFIX_FIT_DID_NOT_COMPLETE_ALL_20_EPOCHS_AND_2000_ADAM')
        best=max(epochs,key=lambda row:(row['test']['binary_accuracy'],-row['epoch']))
        result=dict(role=ROLE,stage='PREFIX_CORRECTNESS_WARMUP20_COMPLETE',config=FIT,binding=binding,
            actual_head_adam_updates=updates,epoch_history=epochs,minibatch_history=minis,
            final_epoch=20,final_metrics=epochs[-1],final_checkpoint=saved[-1],
            best_test_epoch_diagnostic_only=best['epoch'],best_test_metrics_diagnostic_only=best['test'],
            best_epoch_controls_training_or_final_model=False,actor_updated=False,formal_training_started=False)
        once(output/'metrics.json',result)
        once(output/'ready.json',dict(stage=result['stage'],actual_head_adam_updates=updates,completed_epochs=20,
            epoch_records=21,minibatch_records=2000,final_epoch=20,final_checkpoint=saved[-1],
            metrics_sha256=sha(output/'metrics.json'),binding=binding,best_epoch_diagnostic_only=best['epoch'],
            actor_updated=False,formal_training_started=False,resource_release_audit_required=True))
        return result
    except BaseException:
        partial=checkpoint(output/'partial-head.pt',model,optimizer,rng,epoch=epochs[-1]['epoch'] if epochs else 0,
            updates=updates,binding=binding)
        once(output/'partial-state.json',dict(actual_head_adam_updates=updates,actual_adam_steps=adam_steps(optimizer),
            completed_epoch_records=len(epochs),completed_minibatch_records=len(minis),checkpoint=partial,no_automatic_retry=True))
        raise
    finally:model.train(prior)


def load_runtime(path):
    path=remote(path)
    if (subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()!=RUNTIME_COMMIT or
            subprocess.check_output(['git','-C',str(path),'status','--porcelain']).strip()):
        raise ValueError('PREFIX_FIT_REQUIRES_IMMUTABLE_FCB_RUNTIME')
    sys.path.insert(0,str(path))
    from reproduction.a800_common import qprobe_qwen_block_warmup as helpers
    if Path(helpers.__file__).resolve()!=path/'reproduction/a800_common/qprobe_qwen_block_warmup.py':
        raise ValueError('PREFIX_FIT_RUNTIME_IMPORT_SHADOWED_RUN_AS_STANDALONE_FILE')
    from reproduction.a800_common.deploy import published_source
    mapping=published_source(path,RUNTIME_COMMIT)
    if any(sha(path/name)!=value for name,value in mapping.items()):raise ValueError('PREFIX_FIT_RUNTIME_DISK_SOURCE_CHANGED')
    return helpers,hashlib.sha256(canonical(mapping)).hexdigest()


def run(args):
    if args.epochs!=20 or args.physical_index is None:raise ValueError('PREFIX_FIT_REQUIRES_ALL_20_EPOCHS_AND_EXPLICIT_GPU')
    helpers,runtime_map=load_runtime(args.runtime_repo)
    feature_path=Path(__file__).with_name('qprobe_prefix_correctness_features.py')
    spec=importlib.util.spec_from_file_location('prefix_correctness_features_provider',feature_path)
    provider=importlib.util.module_from_spec(spec);sys.modules[spec.name]=provider;spec.loader.exec_module(provider)
    cache=provider.PrefixFeatureCache(remote(args.cache_dir),expected_manifest_sha=args.cache_manifest_sha256)
    seal=cache.verify_all();ids,positive=validate_cache(cache)
    model_dir=remote(cache.spec['encoder_model_dir']);model_hashes=cache.spec['encoder_files_sha256']
    if model_dir!=Path('/data/search-g1/assets/qwen2.5-3b-instruct') or sha(model_dir/'config.json')!=cache.spec['encoder_config_sha256'] or cache.spec['encoder_config_sha256']!=model_hashes['config.json']:
        raise ValueError('PREFIX_FIT_OFFICIAL_MODEL_SOURCE_CHANGED')
    official=helpers.official_config(model_dir);block_source=helpers.block_index_binding(model_dir,model_hashes)
    output=remote(args.output_dir);output.mkdir(mode=0o700,parents=True,exist_ok=False)
    once(output/'run-claim.json',dict(role=ROLE,pid=os.getpid(),source_sha256=sha(Path(__file__)),cache_manifest_sha256=sha(cache.path/'manifest.json'),
        runtime_commit=RUNTIME_COMMIT,no_automatic_retry=True,claimed_unix=time.time()))
    lock=None
    try:
        born=born_gpu_uuid(args.physical_index)
        lock,gpu=helpers.qualified.gpu_admission(args.physical_index)
        if gpu['physical_gpu_uuid']!=born or gpu['actual_cuda_uuids']!=[born] or os.environ.get('CUDA_VISIBLE_DEVICES')!=born:
            raise ValueError('PREFIX_FIT_GPU_ADMISSION_CHANGED_THE_BORN_UUID')
        import torch
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        from search_g1.qprobe_qwen_block_model import QwenBlockQProbeConfig,native_implementation_binding,_tensor_sha
        config=QwenBlockQProbeConfig(max_tokens=cache.max_tokens,epochs=20,batch_size=256,micro_batch_size=32,learning_rate=.0003,seed=1729)
        state=helpers.load_block_weights(model_dir,block_source)
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(torch.Generator().manual_seed(FIT['seed']).get_state());model=build_probe(config)
        copy_proof=model.copy_official_block(state);helpers.verify_copied_block(copy_proof,block_source);del state
        readout=dict(outputs=1,fresh=True,seed=1729,weight_sha256=_tensor_sha(model.output.weight),bias_sha256=_tensor_sha(model.output.bias))
        native=native_implementation_binding(config)
        binding=dict(runtime_commit=RUNTIME_COMMIT,runtime_source_map_sha256=runtime_map,fit_source_sha256=sha(Path(__file__)),
            feature_provider_source_sha256=sha(feature_path),cache_manifest_sha256=sha(cache.path/'manifest.json'),
            cache_spec_sha256=cache.spec_sha,cache_full_sha_seal=seal,split_proof=cache.manifest['split_proof'],
            positive_counts=positive,global_max_tokens=cache.max_tokens,actual_position_ids_from_cache=True,
            official_model_config=official,official_block_source=block_source,initial_block_proof=copy_proof,
            initial_readout_proof=readout,native_implementation=native,gpu=dict(gpu,born_cuda_visible_devices=born),config=FIT,
            precision=dict(head_parameters='float32',float32_matmul_precision=torch.get_float32_matmul_precision(),
                cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32))
        once(output/'prepared.json',binding);model.to(device='cuda:0',dtype=torch.float32)
        if any(p.device.type!='cuda' or p.dtype!=torch.float32 or not p.requires_grad for p in model.parameters()):
            raise ValueError('PREFIX_FIT_ALL_NATIVE_BLOCK_AND_SINGLE_READOUT_MUST_TRAIN_IN_FP32')
        result=fit(model,cache,ids,output,binding)
        event(output,'all_20_epochs_complete',actual_head_adam_updates=result['actual_head_adam_updates'],final_epoch=20,
            final_test_binary_accuracy=result['final_metrics']['test']['binary_accuracy'],
            final_test_majority_baseline_accuracy=result['final_metrics']['test']['majority_baseline_accuracy'])
        return 0
    except BaseException as failure:
        once(output/'failure.json',dict(failure_type=type(failure).__name__,frames=[dict(file=Path(frame.filename).name,line=frame.lineno,function=frame.name)
            for frame in traceback.extract_tb(failure.__traceback__)[-8:]],partial_state_preserved=True,no_automatic_retry=True,checked_unix=time.time()))
        raise
    finally:
        if lock is not None:lock.close()


def cpu_structure_check():
    """Small real CPU mask/loss/mean-accumulation check; no native model allocation."""
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    if torch.cuda.is_initialized():raise ValueError('PREFIX_FIT_CPU_CHECK_CUDA_ALREADY_INITIALIZED')
    x=torch.arange(24,dtype=torch.float32).reshape(2,3,4).requires_grad_();mask=torch.tensor([[True,True,False],[True,True,True]])
    positions=torch.tensor([[10,11,-1],[20,21,22]],dtype=torch.long)
    clean,actual_mask,actual_positions=prepare_batch(x,mask,positions,hidden_size=4,max_tokens=6,max_positions=32)
    assert not clean.requires_grad and torch.equal(actual_positions[1],positions[1]) and not clean[0,2].any()
    changed=x.detach().clone();changed[0,2]=float('nan')
    assert torch.equal(prepare_batch(changed,mask,positions,hidden_size=4,max_tokens=6,max_positions=32)[0],clean)
    labels=torch.tensor([0.,1.,1.,0.]);logits=torch.tensor([-2.,2.,-2.,2.]);metrics=binary_metrics(logits,labels,training_prior=.5)
    assert metrics['confusion']==dict(tp=1,tn=1,fp=1,fn=1) and metrics['binary_accuracy']==.5 and metrics['binary_bce']==metrics['nll']
    a=torch.nn.Linear(2,1);b=torch.nn.Linear(2,1);b.load_state_dict(a.state_dict())
    inputs=torch.tensor([[1.,2.],[3.,4.],[5.,6.],[7.,8.]])
    torch.nn.functional.binary_cross_entropy_with_logits(a(inputs).squeeze(-1),labels).backward()
    for i in (slice(0,2),slice(2,4)):
        (torch.nn.functional.binary_cross_entropy_with_logits(b(inputs[i]).squeeze(-1),labels[i])*.5).backward()
    assert all(torch.allclose(p.grad,q.grad,rtol=1e-6,atol=1e-7) for p,q in zip(a.parameters(),b.parameters()))
    assert 20*math.ceil(25600/256)==2000 and FIT['class_weights'] is None and FIT['outputs']==1
    assert not torch.cuda.is_initialized()
    print(json.dumps(dict(cpu_mask_loss_accumulation_schedule_checks_passed=True,native_forward_executed=False,
        actual_positions_preserved=True,padding_noncontribution=True,input_grad_detached=True,expected_head_adam_updates=2000)))
    return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-repo',type=Path);parser.add_argument('--cache-dir',type=Path);parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--cache-manifest-sha256');parser.add_argument('--physical-index',type=int);parser.add_argument('--epochs',type=int,default=20)
    parser.add_argument('--cpu-structure-check',action='store_true');args=parser.parse_args()
    if args.cpu_structure_check:return cpu_structure_check()
    if any(value is None for value in (args.runtime_repo,args.cache_dir,args.output_dir,args.physical_index)):
        parser.error('--runtime-repo, --cache-dir, --output-dir and --physical-index are required')
    return run(args)


if __name__=='__main__':raise SystemExit(main())
