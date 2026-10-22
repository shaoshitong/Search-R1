"""Resume E1; joint E2 at5e-7, then frozen-native/head-only E3 at5e-5."""
import argparse,fcntl,gc,json,os,signal,time,traceback
from pathlib import Path
import torch
from . import probe_ablation_train as u,mixed_group_checkpoint as original_ck
from . import mixed_group_phased_checkpoint as ck
from . import joint_mixed_direct_b128 as prior
from .direct_update_phased import direct_update_phased
from .replay_feature_reader_fast import PrefetchedReplayFeatureReader

BASE=prior.source.B/'algorithm-improvement-20261021/mixed-direct-e1-lr-headonly-20261022-v1'
STOP=False

def configure(model,native,head,phase):
    frozen=phase=='phase3_head_only'
    model.block.requires_grad_(not frozen)
    if frozen:
        for p in model.block.parameters():p.grad=None
    for group in native.param_groups:group['lr']=5e-7
    for group in head.param_groups:group['lr']=5e-5 if frozen else 5e-7

def binding(base,c):
    return dict(config_sha256=u.sha(base/'config.json'),matched_binding_sha256=u.digest(c),
                initialization_source='official_initial',source_epoch1_checkpoint=c['source_e1']['source_checkpoint'])

def run(base):
    c=u.read(base/'config.json');spec=c['source_e1'];old=u.read(u.checked(spec['source_config']))
    if c['mode']!='e1_lr_then_headonly' or c['epoch_end_cursors']!=[3898,5219] or c['update_budget']!=5219 or c['batch_questions']!=128:
        raise ValueError('User E1/E2/E3 phased recipe differs')
    for name,sha in c['code_hashes'].items():
        if u.sha(Path(__file__).resolve().parents[3]/name)!=sha:raise ValueError('Unpublished runtime change '+name)
    if not os.environ.get('SLURM_JOB_ID')or not torch.cuda.is_available()or torch.cuda.device_count()!=1:
        raise ValueError('Own one-GPU formal allocation required')
    device=torch.device('cuda',0);torch.cuda.set_device(device);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    for sig in(signal.SIGUSR1,signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:globals().__setitem__('STOP',True))
    u.atomic(base/'controller.json',dict(pid=os.getpid(),job_id=os.environ['SLURM_JOB_ID'],stage='RESTORING_COMPLETE_E1',source_cursor=2577,time=time.time()))
    groups,blocked,hashes=prior.source.own_groups(old);lookup={g['qid']:g for g in groups}
    sampler=prior.B128Sampler(groups,old,blocked,hashes);bind=binding(base,c)
    model,ema,native,head,_=prior.source.initialize(old,device);prior.source.warm_gradient_buckets(model)
    if(base/'latest.json').exists():
        saved=torch.load(u.checked(u.read(base/'latest.json')),map_location='cpu',mmap=True,weights_only=False)
        cursor,seen=ck.restore(saved,model,ema,native,head,device,sampler,bind)
        ps=dict(saved['phase_state']);provenance=saved['official_initialization']
    else:
        saved=torch.load(u.checked(spec['source_checkpoint']),map_location='cpu',mmap=True,weights_only=False)
        cursor,seen=original_ck.restore(saved,model,ema,native,head,device,sampler,prior.binding(old))
        if cursor!=2577:raise ValueError('Exact epoch1 snapshot required')
        provenance=saved['official_initialization']
        ps=dict(training_phase='phase2_joint_low_lr',native_freeze_cursor=3898,
                native_frozen_boundary_sha256=None,source_epoch1_checkpoint=spec['source_checkpoint'])
        u.atomic(base/'source-restore-complete.json',dict(status='COMPLETE_E1_RAW_EMA_TWO_ADAM_RNG_SAMPLER_RESTORED',
                 source_checkpoint=spec['source_checkpoint'],source_cursor=cursor,source_accepted_hashes=saved['accepted_hashes'],
                 seen_questions=len(seen),raw_replaced_by_EMA=False,sampler_binding_and_position_preserved=True,
                 next_native_lr=5e-7,next_head_lr=5e-7,epoch_time=time.time()))
    del saved;gc.collect();configure(model,native,head,ps['training_phase'])
    inherited=u.read(u.checked(spec['reader_integrity']))
    seal={k:inherited[k]for k in('source_config','source_checkpoint','restoration_receipt')}
    reader=PrefetchedReplayFeatureReader.from_inherited_seal(u.read(u.checked(c['features'])),26,seal)
    reader.proof['immediate_restored_checkpoint']=spec['source_checkpoint'];u.atomic(base/'reader-integrity.json',reader.proof)
    micro=u.read(base/'progress.json').get('micro_questions',64) if(base/'progress.json').exists() else 64
    def epoch_due():return {3898:2,5219:3}.get(cursor)
    def save():
        return ck.save(base,ck.snapshot(model,ema,native,head,device,sampler,cursor,bind,provenance,seen,
                    'evaluation_due'if epoch_due()else'training',phase_state=ps),pin=cursor==2577 or epoch_due()is not None)
    save()
    try:
        while cursor<5219 and not STOP:
            epoch=epoch_due()
            if epoch and not(base/'evaluations'/('epoch-%02d'%epoch)/'complete.json').exists():
                reader.reset_prefetch();prior.evaluate(base,c,model,ema,native,head,device,sampler,provenance,save(),epoch)
            if cursor>=3898 and ps['training_phase']!='phase3_head_only':
                ps=dict(ps,training_phase='phase3_head_only',native_frozen_boundary_sha256=ck.native_hash(model))
                configure(model,native,head,ps['training_phase'])
                u.atomic(base/'phase-transition.json',dict(cursor=cursor,phase_state=ps,native_steps=original_ck.optimizer_steps(native.state_dict(),True),
                         head_steps=original_ck.optimizer_steps(head.state_dict(),True),native_lr=5e-7,head_lr=5e-5,
                         raw_native_frozen=True,EMA_full_model_passive_tracking=True,time=time.time()))
            reader.prepare(sampler);u.atomic(base/'update-intent.json',dict(cursor=cursor+1,phase=ps['training_phase'],state='started',time=time.time()))
            metrics,batch=direct_update_phased(model,ema,native,head,reader,lookup,sampler,device,c['ema_decay'],micro_questions=micro,phase=ps['training_phase'])
            cursor+=1;micro=metrics['micro_questions'];seen.update(batch['source_qids'])
            row=dict(cursor=cursor,budget=5219,source_cursor=2577,new_updates=cursor-2577,
                     question_visits_since_E1=(cursor-2577)*128,seen_questions=len(seen),time=time.time(),
                     training_phase=ps['training_phase'],prefetch=dict(getattr(reader,'prefetch_statistics',{})),**metrics)
            u.atomic(base/'update-intent.json',dict(cursor=cursor,phase=ps['training_phase'],state='accepted_update_complete'))
            u.atomic(base/'progress.json',row)
            with(base/'curve.jsonl').open('a')as f:f.write(json.dumps(row)+'\n');f.flush()
            print(json.dumps(row),flush=True)
            if cursor in(2578,3899)or cursor%128==0 or epoch_due():save()
        receipt=save();epoch=epoch_due()
        if epoch and not STOP:
            reader.reset_prefetch();prior.evaluate(base,c,model,ema,native,head,device,sampler,provenance,receipt,epoch)
        result=dict(status='INTERRUPTED_AFTER_COMPLETE_ACCEPTED_UPDATE'if STOP else'E1_CONTINUATION_LOW_LR_E2_HEADONLY_E3_COMPLETE',
                    cursor=cursor,native_steps=original_ck.optimizer_steps(native.state_dict(),True),
                    head_steps=original_ck.optimizer_steps(head.state_dict(),True),phase_state=ps,checkpoint=receipt,
                    global_batch_questions=128,source_cursor=2577,seen_questions=len(seen),
                    final4096_scored=False,actor_updates=0)
        u.atomic(base/('interrupted.json'if STOP else'complete.json'),result);return result
    finally:reader.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--action',choices=('train',),default='train');a=p.parse_args();base=a.base.resolve()
    if base!=BASE:p.error('Independent E1 LR/freeze namespace required')
    with(base/'training.lock').open('a+')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:print(json.dumps(run(base)),flush=True)
        except BaseException as e:
            u.atomic(base/'failure.json',dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc(),time=time.time()));raise
if __name__=='__main__':main()
