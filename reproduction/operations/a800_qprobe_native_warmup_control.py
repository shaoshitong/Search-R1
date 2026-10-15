"""One remote GPU, original Head200/original5120 labels/features, inference only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode=True
HEAD_SHA='9b32f40ea573b855dc9e3be627fce7be1c03a1ba0b0e374808a81db311bb6849'
WARMUP=Path('/data/shaoshitong/Search-G1-qprobe-qwen-lastblock-warmup10-096c8297cba74f8089063e763686671e/warmup')
TEACHER=Path('/data/shaoshitong/Search-G1-qprobe-shared/teacher-warmup-v4-sigma3-fullouter')


def read(p):return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for part in iter(lambda:f.read(8*1024**2),b''):h.update(part)
    return h.hexdigest()


def save(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runtime-repo',required=True)
    p.add_argument('--output-dir',required=True);p.add_argument('--physical-index',type=int,required=True);a=p.parse_args()
    repo=Path(a.runtime_repo);out=Path(a.output_dir)
    assert repo.is_absolute() and repo.resolve()==repo and out.is_absolute() and out.resolve()==out
    assert str(out).startswith(('/data/','/mnt/','/backup/')) and not out.is_relative_to(repo)
    assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()=='fcb77204f48d5cb8ec21c68df7d4a2c937b3557a'
    assert a.physical_index in (0,1,2,3,5,6)
    out.mkdir(parents=True,exist_ok=False);os.environ['CUDA_VISIBLE_DEVICES']=str(a.physical_index)
    sys.path.insert(0,str(repo))
    from reproduction.a800_common.qprobe_token_warmup import gpu_admission
    lock,gpu=gpu_admission(a.physical_index);save(out/'gpu-admission.json',gpu)
    import torch
    from search_g1.qprobe_qwen_block_model import QwenBlockQProbeConfig,QwenLastBlockQuestionProbe,QwenBlockWarmup
    from search_g1.qprobe_token_model import TokenBinaryWarmup,token_binary_loss
    from search_g1.qprobe_token_features import TokenFeatureCache
    from search_g1.qprobe_warmup import moments
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    ready=read(WARMUP/'ready.json');assert ready['head_sha256']==HEAD_SHA==sha(WARMUP/'fresh-head.pt')
    raw=torch.load(WARMUP/'fresh-head.pt',map_location='cpu',weights_only=False);head=raw['head']
    assert head['update_count']==200 and head['fitted'] is True
    cache=TokenFeatureCache(ready['feature_cache_dir'],expected_spec_sha=ready['token_feature_spec_sha256'])
    complete=read(TEACHER/'complete.json');groups={}
    assert sha(TEACHER/'identity.json')==raw['label_producer_provenance']['teacher_identity_sha256']
    assert sha(TEACHER/'complete.json')==raw['label_producer_provenance']['teacher_complete_sha256']
    for part in complete['chunks']:
        path=TEACHER/part['relative_path'];assert path.parent==TEACHER and sha(path)==part['sha256']
        for g in read(path)['groups']:
            assert g['qid'] not in groups and moments(g['strict_em'])[0]==g['correct_count'];groups[g['qid']]=g
    ids=cache.ids;assert len(ids)==len(groups)==5120 and set(ids)==set(groups)
    assert all(cache.index[q]['question_hash']==groups[q]['question_hash'] for q in ids)
    counts=torch.tensor([groups[q]['correct_count'] for q in ids],dtype=torch.long)
    lengths=torch.tensor([cache.index[q]['length'] for q in ids],dtype=torch.long)
    config=QwenBlockQProbeConfig(**head['config']);assert config.micro_batch_size==32 and config.max_tokens==244
    model=QwenLastBlockQuestionProbe(config);model.load_state_dict(head['model'],strict=True)
    model.requires_grad_(False);model.to(device='cuda:0',dtype=torch.float32);model.eval()
    loader=QwenBlockWarmup.__new__(QwenBlockWarmup);loader.config=config;loader.device=torch.device('cuda:0')
    values={k:[] for k in ('binary_logit','mixed_probability','predicted_statistic')}
    sums={k:0. for k in ('total_loss','binary_bce','regression_mse_raw','regression_mse_normalized','regression_contribution')}
    started=time.monotonic();done=0
    with torch.inference_mode():
        for index in torch.arange(5120).split(32):
            features,sizes=TokenBinaryWarmup._batch(loader,cache,index,lengths,ids);pred=model(features,sizes)
            loss,parts=token_binary_loss(pred,counts[index],config)
            for k,v in dict(total_loss=loss,**parts).items():sums[k]+=float(v)*len(index)
            for k in values:values[k].append(pred[k].detach().cpu())
            done+=len(index)
            if done%256==0:print(json.dumps(dict(stage='ORIGINAL5120_HEAD200_INFERENCE',completed=done,total=5120,elapsed_seconds=time.monotonic()-started)),flush=True)
    values={k:torch.cat(v) for k,v in values.items()};prob=values['mixed_probability'];truth=(counts>0)&(counts<5);positive=prob>=.5
    tp,tn,fp,fn=[int(v.sum()) for v in (truth&positive,~truth&~positive,~truth&positive,truth&~positive)]
    result=dict(role='original5120_warmup_head200_inference_control',questions=5120,correct=tp+tn,binary_accuracy=(tp+tn)/5120,
        expected_original_correct=5114,reproduces_original_5114_of_5120=tp+tn==5114,
        binary_confusion=dict(true_positive=tp,true_negative=tn,false_positive=fp,false_negative=fn),
        true_mixed_fraction=float(truth.float().mean()),predicted_mixed_fraction=float(positive.float().mean()),
        probability_range=[float(prob.min()),float(prob.max())],logit_range=[float(values['binary_logit'].min()),float(values['binary_logit'].max())],
        probability_exact_zero_count=int((prob==0).sum()),probability_exact_one_count=int((prob==1).sum()),
        **{k:v/5120 for k,v in sums.items()},loss_identity=config.loss_identity,head_sha256=HEAD_SHA,
        cache_dir=str(cache.path),cache_spec_sha256=cache.spec_sha,model_eval=True,micro_batch_size=32,
        global_max_tokens=244,compute_dtype='float32',allow_tf32=False,head_optimizer_updates=0,actor_optimizer_updates=0,
        new_random512_test=False,elapsed_seconds=time.monotonic()-started,checked_unix=time.time(),gpu=gpu)
    save(out/'predictions.json',[dict(qid=q,correct_count=int(counts[i]),**{k:float(v[i]) for k,v in values.items()}) for i,q in enumerate(ids)])
    save(out/'summary.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
