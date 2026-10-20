"""Isolated CP2 continuation: unique true-Mixed groups, LR5e-6, supervised rank aid.

Full raw/EMA/Adam/four-rank RNG continuation. No actor updates, resampling,
question-selection oracle, feature mutation, or use of diagnostic data for fit.
"""
import argparse,copy,gzip,hashlib,json,math,os,random,signal,time
from pathlib import Path
from datetime import timedelta
from contextlib import nullcontext
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from .download_model import atomic,fingerprints
from .packed_features import FeatureReader
from .model import Probe,native_buffers,restore_buffers
from .train import model_hash,optimizer_hash,actual_steps,restore_state,evaluate
from .mixed_loss import mixed_only_loss
STOP=False

def stop(_sig,_frame):
    global STOP
    STOP=True

def read(path):return json.loads(Path(path).read_text())

def plan(n,seed,epoch):
    ids=list(range(n));random.Random(seed+epoch*100003).shuffle(ids)
    batches=[ids[i:i+51] for i in range(0,n,51)]
    if len(batches[-1])<4:
        needed=4-len(batches[-1]);batches[-1][:0]=batches[-2][-needed:];del batches[-2][-needed:]
    assert sorted(i for b in batches for i in b)==list(range(n)) and min(map(len,batches))>=4
    return batches

def update(ddp,model,ema,optimizer,reader,groups,indices,rank,device,c,commit=True):
    own=indices[rank::4];optimizer.zero_grad(set_to_none=True);before=actual_steps(optimizer)
    stats=torch.zeros(10,dtype=torch.float64,device=device);started=time.monotonic()
    for offset in range(0,len(own),2):
        part=[groups[i] for i in own[offset:offset+2]];n=len(part)
        data=reader.get([tid for g in part for tid in g['trajectory_ids']],device)
        labels=torch.tensor([g['labels'] for g in part],dtype=torch.float32,device=device)
        masks=torch.tensor([g['pair_mask'] for g in part],dtype=torch.bool,device=device)
        with (ddp.no_sync() if offset+2<len(own) else nullcontext()):
            with torch.autocast('cuda',dtype=torch.bfloat16):z=ddp(data).reshape(n,5)
            loss=mixed_only_loss(z,labels,masks,c['pair_weight'],c['pair_margin'])
            (loss['loss']*(4*n/len(indices))).backward()
        stats+=torch.tensor([n,n*float(loss['trajectory_bce'].detach()),n*float(loss['group_bce'].detach()),n*float(loss['pairwise_loss'].detach()),n*float(loss['loss'].detach()),float(loss['eligible_pairs']),float(loss['pairwise_correct']),float(loss['eligible_groups']),n*float(loss['mean_p_range'].detach()),0.],device=device,dtype=torch.float64)
    finite=torch.tensor(int(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())),device=device)
    dist.all_reduce(finite,op=dist.ReduceOp.MIN);assert finite.item()==1,'Nonfinite probe gradient'
    if commit:
        optimizer.step();assert actual_steps(optimizer)==before+1
        with torch.no_grad():
            for target,value in zip(ema.parameters(),model.parameters()):target.mul_(.99).add_(value,alpha=.01)
    else:
        assert actual_steps(optimizer)==before,'Reducer warmup must not perform an optimizer update'
    stats[9]=time.monotonic()-started;dist.all_reduce(stats);n=float(stats[0])
    return dict(trajectory_bce=float(stats[1]/n),group_bce=float(stats[2]/n),pairwise_loss=float(stats[3]/n),total_loss=float(stats[4]/n),pairwise_accuracy=float(stats[6]/stats[5]) if stats[5] else None,eligible_pairs=int(stats[5]),eligible_groups=int(stats[7]),mean_p_range=float(stats[8]/n),mean_rank_step_seconds=float(stats[9]/4))

@torch.no_grad()
def separation(model,reader,groups,rank,device,c):
    model.eval();s=torch.zeros(10,dtype=torch.float64,device=device);own=groups[rank::4]
    for start in range(0,len(own),8):
        part=own[start:start+8];n=len(part)
        data=reader.get([tid for g in part for tid in g['trajectory_ids']],device)
        y=torch.tensor([g['labels'] for g in part],device=device,dtype=torch.float32)
        mask=torch.tensor([g['pair_mask'] for g in part],device=device,dtype=torch.bool)
        with torch.autocast('cuda',dtype=torch.bfloat16):z=model(data).reshape(n,5)
        loss=mixed_only_loss(z,y,mask,c['pair_weight'],c['pair_margin']);p=z.sigmoid();ran=p.amax(1)-p.amin(1)
        s+=torch.tensor([n,n*float(loss['trajectory_bce']),n*float(loss['group_bce']),n*float(loss['pairwise_loss']),float(loss['eligible_pairs']),float(loss['pairwise_correct']),float((ran<=.01).sum()),float(((p>=.4)&(p<=.6)).all(1).sum()),float(ran.sum()),float(((p>=.5)==y.bool()).sum())],device=device,dtype=torch.float64)
    dist.all_reduce(s);n=float(s[0])
    return dict(questions=int(n),trajectory_bce=float(s[1]/n),group_bce=float(s[2]/n),pairwise_loss=float(s[3]/n),eligible_pairs=int(s[4]),pairwise_accuracy=float(s[5]/s[4]) if s[4] else None,range_le001_fraction=float(s[6]/n),all_p_in04_06_fraction=float(s[7]/n),mean_p_range=float(s[8]/n),trajectory_accuracy=float(s[9]/(5*n)))

@torch.no_grad()
def selection(model,reader,groups,rank,device,path):
    from .heldout_selection import mixed_probability,assess
    model.eval();out=[];own=groups[rank::4]
    for start in range(0,len(own),8):
        part=own[start:start+8]
        data=reader.get([tid for g in part for tid in g['trajectory_ids']],device)
        with torch.autocast('cuda',dtype=torch.bfloat16):z=model(data).reshape(len(part),5)
        for row,p in zip(part,z.sigmoid().cpu().tolist()):out.append(dict(qid=row['qid'],trajectory_ids=row['trajectory_ids'],p_j=p,predicted_mixed=mixed_probability(p)))
    gathered=[None]*4;dist.all_gather_object(gathered,out)
    result=None
    if rank==0:
        predictions=[p for own in gathered for p in own];assert len(predictions)==2048 and len({p['qid'] for p in predictions})==2048
        selected=sorted(predictions,key=lambda p:(-p['predicted_mixed'],p['qid']))[:512]
        truth={g['qid']:g['labels'] for g in groups}
        temp=path.with_suffix('.tmp')
        with gzip.open(temp,'wt') as f:
            for p in predictions:f.write(json.dumps(p)+'\n')
        temp.replace(path)
        result=dict(pool=assess(predictions,truth),selected512=assess(selected,truth),predictions=dict(path=str(path),**fingerprints(path)),evaluation_only=True)
    result_list=[result];dist.broadcast_object_list(result_list,src=0);return result_list[0]

def save(base,model,ema,optimizer,epoch,cursor,steps,metrics,rank,c):
    rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state())
    own=dict(rank=rank,model_sha256=model_hash(model),ema_sha256=model_hash(ema),optimizer_sha256=optimizer_hash(optimizer),steps=actual_steps(optimizer),rng=rng)
    states=[None]*4;dist.all_gather_object(states,own)
    for key in ('model_sha256','ema_sha256','optimizer_sha256'):assert len({s[key] for s in states})==1
    assert all(s['steps']==steps for s in states)
    root=base/'probe';completed=cursor==0
    if rank==0:
        dest=root/('epoch-%04d.pt'%epoch if completed else 'inflight.pt');tmp=dest.with_suffix('.tmp')
        state=dict(role='CP2_MIXED_ONLY_SUPERVISED_PAIR_CONTINUATION',epoch=epoch,cursor=cursor,completed_epoch=completed,steps=steps,source_steps=9146,raw=model.state_dict(),ema=ema.state_dict(),optimizer=optimizer.state_dict(),raw_buffers=native_buffers(model),ema_buffers=native_buffers(ema),rank_states=states,metrics=metrics,actor_updates=0,config_sha256=fingerprints(base/'config.json')['sha256'],bank_sha256=fingerprints(base/'bank/groups.json')['sha256'],fit_code_sha256=fingerprints(Path(__file__))['sha256'],source_commit=c['source_commit'],source_checkpoint=c['source_checkpoint'],epoch_time=time.time())
        with tmp.open('wb') as h:torch.save(state,h);h.flush();os.fsync(h.fileno())
        tmp.replace(dest);receipt=dict(path=str(dest),**fingerprints(dest),epoch=epoch,cursor=cursor,steps=steps,source_steps=9146,completed_epoch=completed,actor_updates=0,source_commit=c['source_commit'],metrics=metrics)
        # This separate CPU reload validates actual saved optimizer/RNG/model state.
        chk=torch.load(dest,map_location='cpu',mmap=True,weights_only=False)
        assert chk['steps']==steps and len(chk['rank_states'])==4
        for key in ('raw','ema'):assert all(bool(torch.isfinite(v).all()) for v in chk[key].values())
        assert {int(s['step']) for s in chk['optimizer']['state'].values()}=={steps}
        assert all(bool(torch.isfinite(s[k]).all()) for s in chk['optimizer']['state'].values() for k in ('exp_avg','exp_avg_sq'))
        for s in chk['rank_states']:assert set(s['rng'])=={'python','numpy','torch','cuda'}
        del chk;receipt['independent_cpu_reload']=True;atomic(root/'latest.json',receipt)
        if completed:atomic(root/('epoch-%04d.json'%epoch),receipt)
    dist.barrier()

def restore(saved,model,ema,optimizer,rank,device):
    # Adam.load_state_dict can retain the CPU step tensors by reference.
    # A trial update must never mutate the reusable checkpoint object in RAM.
    isolated=dict(saved);isolated['optimizer']=copy.deepcopy(saved['optimizer'])
    restore_state(model,ema,optimizer,isolated,rank,device)
    assert model_hash(model)==saved['rank_states'][rank]['model_sha256']
    assert model_hash(ema)==saved['rank_states'][rank]['ema_sha256']
    assert optimizer_hash(optimizer)==saved['rank_states'][rank]['optimizer_sha256']
    assert all(g['lr']==5e-6 and g['weight_decay']==0 for g in optimizer.param_groups)

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);a=p.parse_args();base=a.base;c=read(base/'config.json')
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);assert int(os.environ['WORLD_SIZE'])==4
    assert os.environ.get('FLASH_ATTENTION_DETERMINISTIC')=='1'
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.cuda.set_device(local);device=torch.device('cuda',local);dist.init_process_group('nccl',timeout=timedelta(minutes=30))
    for sig in (signal.SIGUSR1,signal.SIGTERM):signal.signal(sig,stop)
    os.environ['QWEN7B_QUESTION_MAX_TOKENS']='273'
    random.seed(c['seed']+rank);np.random.seed(c['seed']+rank);torch.manual_seed(c['seed']+rank)
    groups=read(base/'bank/groups.json');assert len(groups['train'])==67199 and all(0<sum(g['labels'])<5 for g in groups['train'])
    assert fingerprints(base/'bank/groups.json')['sha256']==c['mixed_bank_sha256']
    assert fingerprints(Path(__file__))['sha256']==c['fit_file_sha256']
    source=Path(c['training_base']);reader=FeatureReader(read(source/'bank/features.json'))
    heldout=Path(c['selection_base']);held_groups=read(heldout/'bank/groups.json')['selection_validation'];held_reader=FeatureReader(read(heldout/'bank/features.json'))
    model=Probe(c['model_dir'],273,c['prefix_tokens']).to(device);ddp=DDP(model,device_ids=[local],broadcast_buffers=False);ema=copy.deepcopy(model).eval().requires_grad_(False)
    optimizer=torch.optim.Adam(model.parameters(),lr=5e-6,weight_decay=0);root=base/'probe';pred=root/'predictions';pred.mkdir(exist_ok=True)
    resumed=(root/'latest.json').exists()
    receipt=read(root/'latest.json') if resumed else c['source_checkpoint']
    assert fingerprints(receipt['path'])['sha256']==receipt['sha256']
    saved=torch.load(receipt['path'],map_location='cpu',mmap=True,weights_only=False)
    if resumed:
        assert saved['config_sha256']==fingerprints(base/'config.json')['sha256'] and saved['fit_code_sha256']==c['fit_file_sha256']
        epoch=saved['epoch']+int(saved['completed_epoch']);cursor=saved['cursor'];steps=saved['steps']
    else:
        assert saved['steps']==9146 and saved['epoch']==2 and saved['completed_epoch'] and saved['actor_updates']==0
        assert saved['config_sha256']==fingerprints(source/'config.json')['sha256'] and saved['bank_sha256']==fingerprints(source/'bank/complete.json')['sha256']
        epoch=1;cursor=0;steps=9146
    migration=None
    if not resumed and c['prefix_tokens']==20:
        from .prefix_migration import extend
        saved,migration=extend(saved,[name for name,_ in model.named_parameters()])
    restore(saved,model,ema,optimizer,rank,device)
    dist.barrier()
    if rank==0:atomic(root/'restore-acceptance.json',dict(status='ACTUAL_FOUR_RANK_RAW_EMA_ADAM_RNG_RESTORED',source_checkpoint=receipt,adam_steps=steps,lr=5e-6,epoch_time=time.time(),new_objective=True,prefix_tokens=c['prefix_tokens'],prefix_migration=migration))
    # Stabilize DDP first-backward bucket layout without changing parameters,
    # EMA, Adam, cursor, or accepted update count. Restore RNG/buffers afterward.
    warm_batch=plan(len(groups['train']),c['seed'],epoch)[0 if cursor>=len(plan(len(groups['train']),c['seed'],epoch)) else cursor]
    model.train();update(ddp,model,ema,optimizer,reader,groups['train'],warm_batch,rank,device,c,commit=False)
    restore(saved,model,ema,optimizer,rank,device)
    dist.barrier()
    if rank==0:atomic(root/'deterministic-warmup.json',dict(status='DDP_GRADIENT_BUCKET_WARMUP_NO_OPTIMIZER_UPDATE',steps=steps,flash_attention_deterministic=True,cublas_workspace=':4096:8',epoch_time=time.time()))
    if not resumed:
        # Actual production CP2 next update is executed twice from restored state;
        # equality is checked, then the second identical result is continued once.
        first=plan(len(groups['train']),c['seed'],1)[0]
        if c['prefix_tokens']==20:
            from .prefix_migration import qualify
            raw_check=qualify(model,reader,[groups['train'][i] for i in first],device)
            ema_check=qualify(ema,reader,[groups['train'][i] for i in first],device)
            all_checks=[None]*4;dist.all_gather_object(all_checks,dict(rank=rank,raw=raw_check,ema=ema_check))
            if rank==0:atomic(root/'prefix20-initialization-proof.json',dict(migration=migration,ranks=all_checks,source_steps=9146))
            restore(saved,model,ema,optimizer,rank,device)
        model.train()
        update(ddp,model,ema,optimizer,reader,groups['train'],first,rank,device,c)
        expected=(model_hash(model),model_hash(ema),optimizer_hash(optimizer))
        from .replay_diagnostics import snapshot,compare
        reference_snapshot=snapshot(model,ema,optimizer)
        restore(saved,model,ema,optimizer,rank,device);model.train()
        initial=update(ddp,model,ema,optimizer,reader,groups['train'],first,rank,device,c)
        actual=(model_hash(model),model_hash(ema),optimizer_hash(optimizer))
        if actual!=expected:
            differences=compare(reference_snapshot,snapshot(model,ema,optimizer))
            atomic(root/('next-update-replay-diagnostics-rank%d.json'%rank),dict(rank=rank,expected_hashes=expected,actual_hashes=actual,differences=differences,deterministic_attention=True,DDP_gradient_warmup=True))
        del reference_snapshot
        exact=torch.tensor(int(actual==expected),device=device);dist.all_reduce(exact,op=dist.ReduceOp.MIN)
        assert exact.item()==1,'CP2 next-update replay mismatch; exact tensor diagnostics saved'
        cursor=1;steps=9147
        proof_ranks=[None]*4;dist.all_gather_object(proof_ranks,dict(rank=rank,raw=actual[0],ema=actual[1],adam=actual[2],replayed_bit_exact=True))
        if rank==0:
            atomic(root/'next-update-proof.json',dict(status='ACTUAL_CP2_FOUR_RANK_NEW_LOSS_NEXT_UPDATE_BIT_EXACT',source_steps=9146,continuation_steps=steps,raw_sha256=actual[0],ema_sha256=actual[1],adam_sha256=actual[2],rank_count=4,rank_proofs=proof_ranks,epoch_time=time.time()))
            atomic(root/'progress.json',dict(epoch=1,batch=1,batches=math.ceil(len(groups['train'])/51),steps=steps,**initial,epoch_time=time.time()))
    del saved
    if not resumed:save(base,model,ema,optimizer,1,1,steps,None,rank,c)
    for e in range(epoch,c['fit_epochs']+1):
        batches=plan(len(groups['train']),c['seed'],e);model.train()
        for index,indices in enumerate(batches):
            if e==epoch and index<cursor:continue
            metrics=update(ddp,model,ema,optimizer,reader,groups['train'],indices,rank,device,c);steps+=1
            if rank==0:
                row=dict(epoch=e,batch=index+1,batches=len(batches),steps=steps,source_steps=9146,**metrics,actor_updates=0,epoch_time=time.time())
                with (root/'curve.jsonl').open('a') as f:f.write(json.dumps(row)+'\n');f.flush()
                atomic(root/'progress.json',row)
            flag=torch.tensor(int(STOP),device=device);dist.all_reduce(flag,op=dist.ReduceOp.MAX)
            if (index+1)%200==0 or flag.item():save(base,model,ema,optimizer,e,index+1,steps,None,rank,c)
            if flag.item():dist.destroy_process_group();return
        # Save all completed updates before evaluation; cursor=len(batches) means
        # resume must finish this epoch's evaluation instead of skipping it.
        save(base,model,ema,optimizer,e,len(batches),steps,None,rank,c)
        if rank==0:atomic(root/'phase.json',dict(epoch=e,phase='student_frozen_probe_epoch_updated_evaluation_pending',steps=steps,epoch_time=time.time()))
        metrics={s:evaluate(ema,reader,groups[s],rank,device,pred,e,s) for s in ('train','diagnostic')}
        metrics['train_separation']=separation(ema,reader,groups['train'],rank,device,c)
        metrics['diagnostic_mixed_separation']=separation(ema,reader,groups['diagnostic_mixed'],rank,device,c)
        metrics['selection2048']=selection(ema,held_reader,held_groups,rank,device,pred/('epoch-%04d-unseen2048.jsonl.gz'%e))
        save(base,model,ema,optimizer,e,0,steps,metrics,rank,c)
        if rank==0:atomic(root/'metrics-latest.json',dict(epoch=e,steps=steps,metrics=metrics,epoch_time=time.time()))
    if rank==0:atomic(root/'complete.json',dict(status='CP2_MIXED_ONLY_FINETUNE_COMPLETE',additional_epochs=c['fit_epochs'],source_steps=9146,steps=steps,latest=read(root/'latest.json'),actor_updates=0,epoch_time=time.time()))
    dist.destroy_process_group()

if __name__=='__main__':main()
