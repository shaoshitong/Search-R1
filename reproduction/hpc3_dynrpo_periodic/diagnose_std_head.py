"""Read-only phase0 std-head diagnosis on <=8 real cached questions, CPU only.

Execute as an independent two-CPU step of this task's existing allocation.
Never run on login, attach to the live trainer, construct Adam, or write tensors.
The accepted head is mmap-loaded; only its model tensors are materialized.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import time

PREFIX = Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-periodic25-thread5-20261016')
CACHE = Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-native-thread5-20261015/warmup-token-features')
TEACHER = Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-sigma3-20261015/teacher-warmup-cache')


def need(condition,message):
    if not condition:raise ValueError('STD_DIAGNOSTIC_'+message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def identity(path):
    st=Path(path).stat()
    return dict(inode=st.st_ino,bytes=st.st_size,mtime_ns=st.st_mtime_ns)


def model_sha(model):
    import torch
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        need(value.device.type=='cpu','MODEL_MUST_STAY_ON_CPU')
        value=value.detach().contiguous()
        h.update(name.encode());h.update(str(value.dtype).encode());h.update(str(tuple(value.shape)).encode())
        h.update(memoryview(value.view(torch.uint8).numpy()).cast('B'))
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--base',required=True,type=Path)
    parser.add_argument('--job',required=True)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--questions',type=int,default=8)
    args=parser.parse_args();base=args.base
    need(base.is_dir() and base.resolve()==base and not base.is_symlink() and
         base.stat().st_uid==os.getuid() and re.fullmatch(re.escape(str(PREFIX))+r'-r[1-9][0-9]*',str(base)),
         'OWN_TASK_NAMESPACE_REQUIRED')
    need(args.job.isdecimal() and os.environ.get('SLURM_JOB_ID')==args.job and
         os.environ.get('SLURM_STEP_ID') not in (None,'batch','extern'),'EXISTING_ALLOCATION_CPU_STEP_REQUIRED')
    need(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and 1<=args.questions<=8 and
         int(os.environ.get('SLURM_CPUS_PER_TASK','0'))==2,'CPU_ONLY_TWO_THREADS_MAX8_REQUIRED')
    need(args.output.is_absolute() and args.output.resolve().is_relative_to(base/'diagnostics') and
         args.output.parent.is_dir() and not args.output.exists(),'OWN_EXCLUSIVE_SMALL_JSON_OUTPUT_REQUIRED')
    run=json.loads((base/'logs/run-manifest.json').read_text())
    allocation=json.loads((base/'logs/native-allocation-admission.json').read_text())
    need(allocation['accepted'] is True and allocation['run_id']==run['run_id'] and
         allocation['slurm_job_id']==args.job and socket.gethostname().split('.')[0]==allocation['node'],
         'ACTUAL_TASK_ALLOCATION_BINDING_CHANGED')
    producer='search_g1/qprobe_qwen_block_model.py'
    source=base/'source'/run['source_commit']/producer
    need(sha(source)==run['source_sha256'][producer],'PINNED_PROBE_SOURCE_SHA_CHANGED')
    os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2',
                      TOKENIZERS_PARALLELISM='false',RAYON_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    need(not torch.cuda.is_initialized(),'CUDA_MUST_NOT_BE_INITIALIZED')
    from search_g1.qprobe_qwen_block_model import QwenBlockQProbeConfig,QwenLastBlockQuestionProbe
    from search_g1.qprobe_token_features import TokenFeatureCache
    ready=json.loads((PREFIX/'warmup/ready.json').read_text())
    head=PREFIX/'warmup/continued-head.pt';head_before=identity(head)
    need(ready['head_adam_steps']==400 and ready['fit_epochs']==20 and
         sha(head)==ready['head_sha256'],'ACCEPTED_PHASE0_HEAD_SHA_CHANGED')
    saved=torch.load(head,mmap=True,map_location='cpu',weights_only=True)
    state=saved['head'];need(state['update_count']==400,'PHASE0_PROBE_UPDATE_COUNT_CHANGED')
    config=QwenBlockQProbeConfig(**state['config']);rng_before=torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):model=QwenLastBlockQuestionProbe(config)
    model.load_state_dict(state['model'],strict=True);model.eval();model.requires_grad_(False)
    need(torch.equal(rng_before,torch.get_rng_state()),'DIAGNOSTIC_CONSTRUCTION_ADVANCED_RNG')
    weights_before=model_sha(model)
    cache=TokenFeatureCache(CACHE,expected_spec_sha=ready['token_feature_spec_sha256'])
    need(sha(CACHE/'manifest.json')==ready['token_feature_manifest_sha256'],'INITIAL_CACHE_MANIFEST_CHANGED')
    complete=json.loads((TEACHER/'complete.json').read_text());chosen=[];seen=set();by_count={}
    for item in complete['chunks']:
        path=TEACHER/item['relative_path'];raw=path.read_bytes()
        need(hashlib.sha256(raw).hexdigest()==item['sha256'],'TEACHER_LABEL_CHUNK_SHA_CHANGED')
        for group in json.loads(raw)['groups']:
            count=group['correct_count'];need(count==sum(group['strict_em']) and 0<=count<=5,'ACTUAL_G5_LABEL_CHANGED')
            if count not in by_count:by_count[count]=(group['qid'],count)
            if len(chosen)<args.questions and group['qid'] not in seen:
                chosen.append((group['qid'],count));seen.add(group['qid'])
    strata=[by_count[k] for k in sorted(by_count)][:args.questions]
    chosen=strata+[(qid,k) for qid,k in chosen if qid not in {q for q,_ in strata}]
    chosen=chosen[:args.questions];rows=[]
    with torch.inference_mode():
        for ordinal,(qid,count) in enumerate(chosen):
            batch=cache.get_batch([qid],device='cpu');length=cache.index[qid]['length']
            prediction=model(batch['features'],torch.tensor([length],dtype=torch.long))
            raw=prediction['raw_outputs'][0];std=prediction['predicted_statistic'][0]
            need(torch.isfinite(raw).all() and torch.isfinite(std),'NONFINITE_REAL_SAMPLE_OUTPUT')
            rows.append(dict(sample_ordinal=ordinal,correct_count=count,target_sample_std=math.sqrt(count*(5-count)/20),
                token_count=length,raw_binary_logit=float(raw[0]),mixed_probability=float(prediction['mixed_probability'][0]),
                raw_std_logit=float(raw[1]),std_sigmoid=float(raw[1].sigmoid()),predicted_sample_std=float(std),
                finite=True,feature_row_sha_verified=qid in cache.verified))
    need(weights_before==model_sha(model),'DIAGNOSTIC_CHANGED_MODEL_WEIGHTS')
    need(torch.equal(rng_before,torch.get_rng_state()),'DIAGNOSTIC_ADVANCED_RNG')
    need(head_before==identity(head),'SOURCE_HEAD_FILE_CHANGED')
    need(not torch.cuda.is_initialized(),'DIAGNOSTIC_INITIALIZED_CUDA')
    logits=[row['raw_std_logit'] for row in rows]
    result=dict(status='READ_ONLY_REAL_PHASE0_STD_HEAD_CPU_DIAGNOSIS',checked_unix=time.time(),
        job=args.job,step=os.environ['SLURM_STEP_ID'],node=socket.gethostname(),pid=os.getpid(),uid=os.getuid(),
        training_source_commit=run['source_commit'],producer_source_sha256=run['source_sha256'][producer],
        diagnostic_source_sha256=sha(__file__),phase0_head_path=str(head),phase0_head_sha256=ready['head_sha256'],
        feature_cache_path=str(CACHE),feature_manifest_sha256=ready['token_feature_manifest_sha256'],
        model_parameter_sha256_before=weights_before,model_parameter_sha256_after=weights_before,
        head_file_identity_unchanged=True,torch_rng_unchanged=True,optimizer_constructed=False,optimizer_state_materialized=False,
        optimizer_metadata_present_in_mmap_container=True,optimizer_tensor_data_accessed=False,mmap_saved_state=True,cuda_initialized=False,
        cpu_intraop_threads=torch.get_num_threads(),cpu_interop_threads=torch.get_num_interop_threads(),
        questions=len(rows),rows=rows,std_logit_min=min(logits),std_logit_max=max(logits),
        std_logit_mean=sum(logits)/len(logits),std_prediction_exact_zero_count=sum(r['predicted_sample_std']==0 for r in rows),
        scope='<=8 stratified actual warmup training questions, diagnosis only; not holdout/AUC or current-student evaluation',
        training_process_touched=False,weights_or_adam_reset=False,raw_questions_or_tensors_output=False)
    with args.output.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    print(json.dumps(result,allow_nan=False),flush=True)


if __name__=='__main__':main()
