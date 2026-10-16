"""Independent CPU clone: exactly one original warmup minibatch, not a restart.

All clone model/Adam/RNG state exists only in remote process RAM. CPU and CUDA
arithmetic can differ; these results are not a bitwise GPU replay.
"""
import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import time

from diagnose_std_head import PREFIX,CACHE,TEACHER,need,sha,identity


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base',required=True,type=Path)
    parser.add_argument('--job',required=True);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();base=args.base;start=time.monotonic()
    result=dict(status='DIAGNOSTIC_PREPARING',job=args.job,original_probe_mutated=False,
                original_adam_loaded=False,clone_optimizer_steps=0,tensors_persisted=False)
    def progress(phase,**values):
        event=dict(phase=phase,elapsed_seconds=time.monotonic()-start,**values);result['last_progress']=event
        path=args.output.with_name('first-adam-progress.json');tmp=path.with_suffix('.partial')
        tmp.write_text(json.dumps(event,allow_nan=False)+'\n');tmp.replace(path)
        print(json.dumps(event,allow_nan=False),flush=True)
    def deadline(*_):raise TimeoutError('Independent diagnostic reached the ten-minute bound')
    signal.signal(signal.SIGALRM,deadline);signal.signal(signal.SIGTERM,deadline);signal.alarm(590)
    try:
        need(base.is_dir() and base.resolve()==base and not base.is_symlink() and base.stat().st_uid==os.getuid() and
             re.fullmatch(re.escape(str(PREFIX))+r'-r[1-9][0-9]*',str(base)),'OWN_TASK_NAMESPACE_REQUIRED')
        need(os.environ.get('SLURM_JOB_ID')==args.job and args.job.isdecimal() and
             os.environ.get('SLURM_STEP_ID') not in (None,'batch','extern') and os.environ.get('SLURM_CPUS_PER_TASK')=='2' and
             os.environ.get('CUDA_VISIBLE_DEVICES')=='','EXISTING_CPU_ONLY_TWO_CPU_STEP_REQUIRED')
        need(args.output.is_absolute() and args.output.resolve().is_relative_to(base/'diagnostics') and
             args.output.parent.is_dir() and not args.output.exists(),'OWN_EXCLUSIVE_SMALL_JSON_REQUIRED')
        run=json.loads((base/'logs/run-manifest.json').read_text());allocation=json.loads((base/'logs/native-allocation-admission.json').read_text())
        need(allocation['accepted'] and allocation['run_id']==run['run_id'] and allocation['slurm_job_id']==args.job and
             socket.gethostname().split('.')[0]==allocation['node'],'ACTUAL_ALLOCATION_BINDING_CHANGED')
        os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',
                          TOKENIZERS_PARALLELISM='false',RAYON_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
        import torch
        torch.set_num_threads(2);torch.set_num_interop_threads(1);need(not torch.cuda.is_initialized(),'CUDA_ALREADY_INITIALIZED')
        from safetensors import safe_open
        from search_g1.qprobe_qwen_block_model import QwenBlockWarmup,QwenBlockQProbeConfig
        from search_g1.qprobe_token_features import TokenFeatureCache
        from search_g1.qprobe_token_model import token_binary_loss
        for name in ('search_g1/qprobe_qwen_block_model.py','search_g1/qprobe_token_model.py','reproduction/hpc3_dynrpo/warmup.py'):
            need(sha(base/'source'/run['source_commit']/name)==run['source_sha256'][name],'BOUND_PRODUCER_SOURCE_CHANGED')
        ready=json.loads((PREFIX/'warmup/ready.json').read_text());original=PREFIX/'warmup/continued-head.pt'
        original_before=identity(original);need(sha(original)==ready['head_sha256'],'ACCEPTED_PHASE0_SHA_CHANGED')
        saved=torch.load(original,map_location='cpu',weights_only=True,mmap=True);state=saved['head']
        need(state['update_count']==400 and len(saved['history']['minibatch_history'])==400,'COMPLETE_PHASE0_HISTORY_REQUIRED')
        config=replace(QwenBlockQProbeConfig(**state['config']),epochs=10)
        model_path=Path(run['initial_model']);index=json.loads((model_path/'model.safetensors.index.json').read_text())['weight_map']
        keys=sorted(k for k in index if k.startswith('model.layers.35.'));need(len(keys)==12,'OFFICIAL_BLOCK35_TENSORS_REQUIRED')
        block={}
        for shard in sorted({index[k] for k in keys}):
            with safe_open(model_path/shard,framework='pt',device='cpu') as stream:
                for key in keys:
                    if index[key]==shard:block[key]=stream.get_tensor(key)
        global_rng=torch.get_rng_state().clone();engine=QwenBlockWarmup(config,block,device='cpu');del block
        need(engine.initial_block_proof==state['initial_block_proof'],'ORIGINAL_OFFICIAL_BLOCK_COPY_PROOF_MISMATCH')
        need(engine.initial_readout_proof==state['initial_readout_proof'],'ORIGINAL_FRESH_READOUT_PROOF_MISMATCH')
        need(engine.native_implementation==state['native_implementation'],'NATIVE_IMPLEMENTATION_MISMATCH')
        need(not engine.optimizer.state and engine.update_count==0 and engine._adam_steps()==0,'CLONE_MUST_HAVE_ZERO_FRESH_ADAM_WRITES')
        need(torch.equal(global_rng,torch.get_rng_state()),'CONSTRUCTION_CHANGED_GLOBAL_RNG')
        cache=TokenFeatureCache(CACHE,expected_spec_sha=ready['token_feature_spec_sha256'])
        need(sha(CACHE/'manifest.json')==ready['token_feature_manifest_sha256'],'INITIAL_FEATURE_MANIFEST_CHANGED')
        complete=json.loads((TEACHER/'complete.json').read_text());labels=[]
        for record in complete['chunks']:
            raw=(TEACHER/record['relative_path']).read_bytes();need(hashlib.sha256(raw).hexdigest()==record['sha256'],'LABEL_CHUNK_CHANGED')
            labels.extend(json.loads(raw)['groups'])
        ids=[g['qid'] for g in labels];need(len(ids)==len(set(ids))==5120,'ORIGINAL5120_ORDER_REQUIRED')
        need(all(cache.index[g['qid']]['question_hash']==g['question_hash'] and g['correct_count']==sum(g['strict_em'])
                 for g in labels),'FEATURE_LABEL_BINDING_CHANGED')
        counts=torch.tensor([g['correct_count'] for g in labels]);lengths=torch.tensor([cache.index[q]['length'] for q in ids])
        reconstruct=torch.Generator(device='cpu').manual_seed(config.seed+1);first=None
        for epoch in range(20):
            permutation=torch.randperm(len(ids),generator=reconstruct)
            if epoch==0:first=permutation.clone()
        need(torch.equal(reconstruct.get_state(),state['private_rng']),'SAVED_PRIVATE_RNG_CANNOT_PROVE_ORIGINAL_PERMUTATIONS')
        order=torch.randperm(len(ids),generator=engine.rng);need(torch.equal(order,first),'CLONE_FIRST_PRIVATE_PERMUTATION_MISMATCH')
        selected=order[:256];need(config.batch_size==256 and config.micro_batch_size==32 and config.seed==1729,'ORIGINAL_BATCH_MICRO_SEED_CHANGED')
        sample=[]
        for count in range(6):
            match=next((i for i,g in enumerate(labels) if g['correct_count']==count),None)
            if match is not None:sample.append(match)
        sample+=[i for i in range(len(ids)) if i not in sample][:8-len(sample)]
        def predict_samples():
            rows=[];prior=engine.model.training;engine.model.eval()
            with torch.inference_mode():
                for ordinal,row in enumerate(sample):
                    x,n=engine._batch(cache,torch.tensor([row]),lengths,ids);p=engine.model(x,n);raw=p['raw_outputs'][0]
                    rows.append(dict(sample_ordinal=ordinal,correct_count=int(counts[row]),
                        target_sample_std=math.sqrt(int(counts[row])*(5-int(counts[row]))/20),raw_std_logit=float(raw[1]),
                        std_sigmoid=float(raw[1].sigmoid()),predicted_sample_std=float(p['predicted_statistic'][0]),
                        raw_binary_logit=float(raw[0]),mixed_probability=float(p['mixed_probability'][0])))
            engine.model.train(prior);return rows
        result.update(initialization_proofs_exact=True,original_private_rng_reconstruction_equal=True,
            initial_readout_proof=engine.initial_readout_proof,original_first_minibatch=saved['history']['minibatch_history'][0],
            first256_question_ids_sha256=hashlib.sha256(json.dumps([ids[i] for i in selected.tolist()],separators=(',',':')).encode()).hexdigest(),
            clone_before=predict_samples(),cpu_affinity=sorted(os.sched_getaffinity(0)))
        progress('INITIALIZATION_AND_ORIGINAL_ORDER_PROVEN',clone_adam_steps=0)
        engine.model.train();engine.optimizer.zero_grad(set_to_none=True)
        scalars=dict(pre_step_loss=0.,binary_bce=0.,regression_mse_raw=0.,regression_mse_normalized=0.,regression_contribution=0.)
        correct=0;raw_grad_squared=0.
        for number,micro in enumerate(selected.split(32),1):
            x,n=engine._batch(cache,micro,lengths,ids);p=engine.model(x,n);p['raw_outputs'].retain_grad()
            loss,parts=token_binary_loss(p,counts[micro],config);fraction=len(micro)/256;(loss*fraction).backward()
            for key,value in dict(pre_step_loss=loss,**parts).items():scalars[key]+=float(value.detach())*fraction
            truth=(counts[micro]>0)&(counts[micro]<5);correct+=int(((p['mixed_probability'].detach()>=.5)==truth).sum())
            raw_grad_squared+=float(p['raw_outputs'].grad[:,1].square().sum())
            need(all(p.grad is None or torch.isfinite(p.grad).all() for p in engine.model.parameters()),'NONFINITE_CLONE_GRADIENT')
            progress('REAL_FIRST_MINIBATCH_BACKWARD_MICRO_COMPLETE',micro=number,of=8,clone_adam_steps=0)
        result['clone_first_minibatch']=dict(scalars,binary_accuracy=correct/256,batch_size=256,micro_batch_size=32,
            std_raw_logit_gradient_l2=math.sqrt(raw_grad_squared),std_readout_weight_gradient_l2=float(engine.model.output.weight.grad[1].norm()),
            std_readout_bias_gradient=float(engine.model.output.bias.grad[1]),binary_readout_weight_gradient_l2=float(engine.model.output.weight.grad[0].norm()),
            binary_readout_bias_gradient=float(engine.model.output.bias.grad[0]))
        reference=result['original_first_minibatch'];result['cpu_minus_original_cuda_first_minibatch']={
            key:result['clone_first_minibatch'][key]-reference[key] for key in list(scalars)+['binary_accuracy']}
        engine.optimizer.step();engine.update_count+=1;result['clone_optimizer_steps']=engine._adam_steps()
        need(result['clone_optimizer_steps']==1 and all(torch.isfinite(p).all() for p in engine.model.parameters()),'CLONE_ONE_ADAM_OR_FINITE_FAILED')
        result['clone_after']=predict_samples()
        need(identity(original)==original_before,'ORIGINAL_HEAD_FILE_CHANGED')
        need(torch.equal(global_rng,torch.get_rng_state()),'DIAGNOSTIC_CHANGED_GLOBAL_RNG')
        need(not torch.cuda.is_initialized(),'DIAGNOSTIC_INITIALIZED_CUDA')
        result.update(status='ACTUAL_ORIGINAL_INIT_AND_FIRST256_ONE_ADAM_CPU_CLONE_COMPLETE',
            original_head_sha256=ready['head_sha256'],original_head_file_identity_unchanged=True,global_torch_rng_unchanged=True,
            original_optimizer_tensor_data_accessed=False,diagnostic_source_sha256=sha(__file__),training_source_commit=run['source_commit'],
            cpu_threads=torch.get_num_threads(),cpu_interop_threads=torch.get_num_interop_threads(),cuda_initialized=False,
            clone_state_persisted=False,scope='One original data minibatch in a CPU clone; CPU/CUDA arithmetic may differ, not bitwise replay')
    except Exception as error:result.update(status='DIAGNOSTIC_LIMITATION_OR_FAILURE',error=repr(error))
    finally:
        signal.alarm(0);result.update(checked_unix=time.time(),elapsed_seconds=time.monotonic()-start,
            pid=os.getpid(),step=os.environ.get('SLURM_STEP_ID'),node=socket.gethostname())
        if args.output.parent.is_dir() and not args.output.exists():
            with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
        print(json.dumps(result,allow_nan=False),flush=True)


if __name__=='__main__':main()
