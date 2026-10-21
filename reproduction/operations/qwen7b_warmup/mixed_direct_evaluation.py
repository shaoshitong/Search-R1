"""Read-only raw/EMA fixed-G5 tests of the direct Mixed logit, never p_j.

The caller first saves an accepted full checkpoint. Evaluation never advances
either Adam, reads the final pool, or invokes the retired trajectory readout.
"""
import fcntl,gzip,hashlib,json,math,os,time
from contextlib import nullcontext
from pathlib import Path
import torch
import torch.nn.functional as F
from . import probe_ablation_train as u


def proof(path):
    p=Path(path);return dict(path=str(p),bytes=p.stat().st_size,sha256=u.sha(p))


def live_state_hashes(model,ema,block_opt,head_opt,device):
    from .mixed_group_checkpoint import tree_hash
    return dict(raw=u.model_mapping_hash(model.state_dict()),ema=u.model_mapping_hash(ema.state_dict()),native_adam=tree_hash(block_opt.state_dict()),head_adam=tree_hash(head_opt.state_dict()),raw_buffers=u.tensor_hash(u.buffers(model)),ema_buffers=u.tensor_hash(u.buffers(ema)),rng=u.rng_hash(u.rng_state(device)))


def selected_qids(rows,top_k=512):
    # No EM/trueMixed read in either ordering key or eligibility.
    return [r['qid']for r in sorted(rows,key=lambda r:(-r['predicted_mixed'],str(r['qid'])))[:top_k]]


def summarize(rows,top_k=512):
    required={'qid','trajectory_ids','strict_em','true_mixed','predicted_mixed'}
    seen=set();tids=set()
    for r in rows:
        if set(r)!=required or r['qid']in seen or len(r['trajectory_ids'])!=5 or len(set(r['trajectory_ids']))!=5 or tids.intersection(r['trajectory_ids']):raise ValueError('Unique authentic complete own G5 direct-score rows required')
        seen.add(r['qid']);tids.update(r['trajectory_ids']);ys=r['strict_em'];q=r['predicted_mixed']
        if len(ys)!=5 or any(type(y)is not int or y not in(0,1)for y in ys)or type(r['true_mixed'])is not bool or r['true_mixed']!=(0<sum(ys)<5)or not math.isfinite(q)or not 0<=q<=1:raise ValueError('Actual own EM and finite direct Mixed probability required')
    if not rows:raise ValueError('No direct Mixed evaluation questions')
    chosen_ids=selected_qids(rows,top_k);by={r['qid']:r for r in rows};chosen=[by[q]for q in chosen_ids]
    pool=u.prediction_metrics(rows);selected=u.prediction_metrics(chosen)
    for value in(pool,selected):
        value['allwrong_false_positive_rate']=value['classwise']['allwrong']['predicted_mixed_fraction']
        value['allcorrect_false_positive_rate']=value['classwise']['allcorrect']['predicted_mixed_fraction']
        value['Mixed_false_negative_rate']=value['classwise']['mixed']['false_negative_rate']
        value['fixed_threshold']=.5
    return dict(pool=pool,selected512=selected,selection_budget=min(top_k,len(rows)),selected_qids_sha256=u.digest(chosen_ids),ordered_qids_sha256=u.digest([r['qid']for r in rows]),ordered_own_trajectory_ids_sha256=u.digest([r['trajectory_ids']for r in rows]),labels_used_to_select=False,threshold_fitted=False,tie_break='descending_direct_Mixed_probability_then_lexicographic_string_qid',score_semantics='sigmoid_of_direct_real_G5_group_logit',trajectory_probability_or_accuracy_produced=False,product_formula_used=False)


def score_reader(model,reader,groups,device,destination,progress,name,micro_groups=1):
    """One forward gives direct group probability and stable binary-logit loss."""
    if not isinstance(micro_groups,int)or not 1<=micro_groups<=4:raise ValueError('Bounded complete G5 microbatch required')
    model.eval();rows=[];sum_loss=0.;classes={key:dict(questions=0,binaryMixedBCE_sum=0.)for key in('allwrong','Mixed','allcorrect')}
    start_time=time.monotonic();io_seconds=compute_seconds=0.
    with torch.inference_mode():
        for start in range(0,len(groups),micro_groups):
            part=groups[start:start+micro_groups]
            if any(len(g['trajectory_ids'])!=5 or len(set(g['trajectory_ids']))!=5 for g in part):raise ValueError('Only authentic own G5 groups')
            before=time.monotonic();batch=reader.get([tid for g in part for tid in g['trajectory_ids']],device)
            if device.type=='cuda':torch.cuda.synchronize(device)
            io_seconds+=time.monotonic()-before;before=time.monotonic()
            with(torch.autocast('cuda',dtype=torch.bfloat16)if device.type=='cuda'else nullcontext()):
                encoded=model.encode_trajectories(batch)
                grouped={key:value.reshape(len(part),5,*value.shape[1:])for key,value in encoded.items()}
                logits=model.group_logits(grouped).float()
            if logits.shape!=(len(part),)or not bool(torch.isfinite(logits).all()):raise ValueError('One finite direct Mixed logit per real G5 required')
            labels=torch.tensor([g['labels']for g in part],device=device,dtype=torch.float32)
            if labels.shape!=(len(part),5)or not bool(((labels==0)|(labels==1)).all()):raise ValueError('Own strict EM5 required')
            counts=labels.sum(-1);target=((counts>0)&(counts<5)).float()
            loss=F.binary_cross_entropy_with_logits(logits,target,reduction='none');probability=logits.sigmoid()
            for g,q,y,value,k in zip(part,probability.cpu().tolist(),target.cpu().tolist(),loss.cpu().tolist(),counts.cpu().tolist()):
                rows.append(dict(qid=g['qid'],trajectory_ids=g['trajectory_ids'],strict_em=g['labels'],true_mixed=bool(y),predicted_mixed=q));sum_loss+=value
                key='allwrong'if k==0 else'allcorrect'if k==5 else'Mixed';classes[key]['questions']+=1;classes[key]['binaryMixedBCE_sum']+=value
            if device.type=='cuda':torch.cuda.synchronize(device)
            compute_seconds+=time.monotonic()-before
            if start%(32*micro_groups)==0 or len(rows)==len(groups):u.atomic(progress,dict(name=name,completed_questions=len(rows),expected_questions=len(groups),completed_trajectories=5*len(rows),elapsed_seconds=time.monotonic()-start_time,feature_io_seconds=io_seconds,compute_seconds=compute_seconds,native_Adam_updates=0,group_head_Adam_updates=0,actor_updates=0,score_semantics='direct_Mixed_sigmoid',observed_epoch=time.time()))
    destination=Path(destination);temporary=destination.with_name(destination.name+'.tmp')
    with gzip.open(temporary,'wt')as f:
        for r in rows:f.write(json.dumps(r,allow_nan=False)+'\n')
    temporary.replace(destination)
    metrics=summarize(rows);metrics['binaryMixedBCE']=sum_loss/len(rows)
    metrics['classwise_binaryMixedBCE']={key:dict(questions=v['questions'],binaryMixedBCE=v['binaryMixedBCE_sum']/v['questions']if v['questions']else None)for key,v in classes.items()}
    metrics['timing']=dict(elapsed_seconds=time.monotonic()-start_time,feature_io_seconds=io_seconds,compute_seconds=compute_seconds)
    return dict(metrics,predictions=proof(destination))


def evaluate(base,c,model,ema,block_opt,head_opt,device,checkpoint,cursor):
    """Accepted full CP first, then raw/EMA fixed512/2048 at integer epochs."""
    if device.type!='cuda'or not os.environ.get('SLURM_JOB_ID'):raise ValueError('Real evaluation requires the owned GPU allocation')
    base=Path(base);steps_per_epoch=c['steps_per_epoch'];config_sha=u.sha(base/'config.json')
    if type(cursor)is not int or cursor<0 or cursor%steps_per_epoch or checkpoint['cursor']!=cursor or checkpoint['native_steps']!=cursor or checkpoint['head_steps']!=cursor or checkpoint['config_sha256']!=config_sha or checkpoint.get('full_CPU_native_head_two_Adam_RNG_sampler_source_reload_passed')is not True:raise ValueError('Save and accept exact full official-source checkpoint before epoch evaluation')
    cp_path=Path(checkpoint['path'])
    if cp_path.parent!=base/'checkpoints'or cp_path.stat().st_size!=checkpoint['bytes']or u.read(base/'latest.json')!=checkpoint:raise ValueError('Exact already accepted current checkpoint receipt required')
    before=live_state_hashes(model,ema,block_opt,head_opt,device)
    if any(checkpoint['accepted_hashes'].get(k)!=v for k,v in before.items()):raise ValueError('Evaluation models/two Adam/RNG/buffers differ from accepted checkpoint')
    epoch=cursor//steps_per_epoch;folder=base/'evaluations'/('epoch-%02d'%epoch);folder.mkdir(parents=True,exist_ok=True);done=folder/'complete.json'
    with(base/'evaluation.lock').open('a+')as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if done.exists():
            prior=u.read(done)
            if prior['checkpoint']!=checkpoint or prior['config_sha256']!=config_sha:raise ValueError('Existing evaluation belongs to another full state')
            for label in('raw','EMA'):
                for value in prior['metrics'][label].values():u.checked(value['predictions'])
            return prior
        from .joint_balanced_native_evaluation import accepted_HF_validation
        from .replay_feature_reader import ReplayFeatureReader
        spec=c['evaluation'];diagnostic=u.read(u.checked(spec['diagnostic_groups']))['diagnostic'];dfeatures=u.read(u.checked(spec['diagnostic_features']))
        if len(diagnostic)!=512:raise ValueError('Actual fixed diagnostic512 required')
        validation,vr,vproof=accepted_HF_validation(spec)
        if len(validation)!=2048:raise ValueError('Actual fixed ownG5 validation2048 required')
        forbidden=u.read(u.checked(c['exclusions']));by={g['qid']:g['question_hash']for g in forbidden['groups']}
        if any(g['qid']not in by or(g.get('question_hash')and by[g['qid']]!=g['question_hash'])for g in diagnostic+validation):raise ValueError('All fixed evaluation questions must be excluded from training')
        dr=ReplayFeatureReader(dfeatures,26);rng=u.rng_state(device);buffers=(u.buffers(model),u.buffers(ema));modes=[[(module,module.training)for module in probe.modules()]for probe in(model,ema)];metrics={}
        try:
            if live_state_hashes(model,ema,block_opt,head_opt,device)!=before:raise ValueError('Reader setup changed accepted state')
            for label,probe in(('raw',model),('EMA',ema)):
                metrics[label]={}
                for name,reader,groups in(('diagnostic512',dr,diagnostic),('fixed_selection2048',vr,validation)):
                    metrics[label][name]=score_reader(probe,reader,groups,device,folder/(label+'-'+name+'.jsonl.gz'),base/'evaluation-progress.json',label+'-'+name,micro_groups=1)
        finally:
            u.restore_rng(rng,device);u.restore_buffers(model,buffers[0]);u.restore_buffers(ema,buffers[1]);dr.close();vr.close()
            for values in modes:
                for module,training in values:module.training=training
        after=live_state_hashes(model,ema,block_opt,head_opt,device)
        if after!=before:raise ValueError('Direct Mixed evaluation changed raw/EMA/two Adam/RNG/native buffers')
        result=dict(status='DIRECT_MIXED_RAW_EMA_FIXED512_2048_EVALUATION_COMPLETE',epoch=epoch,cursor=cursor,checkpoint=checkpoint,config_sha256=config_sha,model_source=dict(official_revision=getattr(model,'official_revision',c.get('official_revision')),official_source_hash=checkpoint['accepted_hashes']['official_source']),metrics=metrics,live_components_before=before,live_components_after=after,models_two_Adam_RNG_native_buffers_unchanged=True,checkpoint_full_SHA_CPU_reload_inherited=True,checkpoint_payload_fresh_SHA_reread=False,checkpoint_saved_before_evaluation=True,actor_updates=0,native_optimizer_updates=0,group_head_optimizer_updates=0,trajectory_probability_or_accuracy_produced=False,product_formula_used=False,threshold_fitted=False,final4096_scored=False,old_internal5120_scored=False,validation_proof=vproof,completed_epoch_time=time.time())
        u.atomic(done,result);return result
