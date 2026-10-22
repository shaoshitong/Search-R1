"""Future one frozen packed-variance candidate versus original E1 EMA.

Preparation/import never opens final journals. Explicit freeze precedes all final
access. Original two2048 ownG5 plans/order are unchanged; candidate gets one
question+five genuine Prefix20 pieces in one packed actor row, as in training.
"""
import argparse,copy,fcntl,gc,gzip,inspect,json,math,os,time,traceback
from pathlib import Path
from . import score_frozen_final4096 as old
B,A,REVISION=old.B,old.A,old.REVISION
BASE=A/'packed-variance-frozen-final4096-20261023-v1'
PREVIOUS_BASES=old.PREVIOUS_BASES+(old.BASE,)
E1=old.E1
THRESHOLD=.08


def freeze_contract(value):
    old.freeze_contract(value)
    c=value['candidate']
    if ((c.get('architecture'),c.get('layer'),c.get('prefix'),c.get('fixed_M'))
            !=('one_question_M3to10_policy20_packed_variance_v1',21,20,5)
            or c.get('state_key')not in('raw','EMA')
            or c.get('selection_score')!='predicted_population_EM_variance'
            or value.get('candidate_variance_threshold')!=THRESHOLD
            or value.get('two_original_2048_G5_plans_unchanged')is not True):
        raise ValueError('Explicit packed21/P20/fixedG5 variance frozen contract required')
    return True


def compact_records(records):
    """Same first20 kept policy tokens as training, not answer/tool suffixes."""
    result={}
    for tid,row in records.items():
        policy=[]
        for segment in row['segments']:
            take=min(20-len(policy),segment['policy_length'])
            policy+=segment['context_tokens'][segment['request_length']:segment['request_length']+take]
            if len(policy)==20:break
        if not row['prompt']or not 0<len(policy)<=20:raise ValueError('Real nonempty Prefix20 required')
        result[tid]=dict(qid=row['qid'],trajectory_id=tid,prompt=row['prompt'],policy20=policy)
    return result


def capture_packed(actor,records,views,separator,device):
    import torch
    from .packed_variance_probe import pack_group
    packed=[pack_group(v,records,separator)for v in views]
    width=max(len(v['input_ids'])for v in packed)
    ids=torch.full((len(views),width),151643,dtype=torch.long,device=device)
    mask=torch.zeros_like(ids,dtype=torch.bool);positions=torch.zeros_like(ids);types=torch.zeros_like(ids)
    for i,row in enumerate(packed):
        n=len(row['input_ids']);ids[i,:n]=torch.tensor(row['input_ids'],device=device)
        mask[i,:n]=torch.tensor(row['attention_mask'],device=device)
        positions[i,:n]=torch.tensor(row['position_ids'],device=device);types[i,:n]=torch.tensor(row['token_types'],device=device)
    with torch.no_grad():features=actor(input_ids=ids,attention_mask=mask,position_ids=positions,use_cache=False,return_dict=True).last_hidden_state.detach()
    return dict(features=features,attention_mask=mask,position_ids=positions,token_types=types)


def score_candidate(descriptor,views,records,device,progress,rank):
    import torch
    from torch import nn
    from transformers import AutoModel,AutoTokenizer
    from .packed_variance_probe import PackedVarianceHead,PackedVarianceProbe
    model_dir=B/'assets/qwen25-7b-instruct-a09a35458c70'
    tok=AutoTokenizer.from_pretrained(str(model_dir),local_files_only=True)
    separator=tok.encode('\n### trajectory\n',add_special_tokens=False)
    if separator!=[198,14374,34682,198]:raise ValueError('Training separator tokenization differs')
    actor=AutoModel.from_pretrained(str(model_dir),local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='flash_attention_2').to(device).eval().requires_grad_(False)
    block=copy.deepcopy(actor.layers[21]).float();rotary=copy.deepcopy(actor.rotary_emb).float()
    actor.layers=nn.ModuleList(list(actor.layers[:21]));actor.norm=nn.Identity();torch.cuda.empty_cache()
    probe=PackedVarianceProbe(block,rotary,'cache_position'in inspect.signature(block.forward).parameters,PackedVarianceHead())
    blob=torch.load(old.checked(descriptor['checkpoint']),map_location='cpu',mmap=True,weights_only=False)
    if ((blob.get('architecture'),blob.get('layer'),blob.get('prefix'))!=('one_question_M3to10_policy20_packed_variance_v1',21,20)
            or blob.get('actor_updates',0)!=0 or blob.get('loss_name')not in('mse','mae','z_mse_anchor')):
        raise ValueError('Exact unchanged official-actor packed variance lineage required')
    probe.load_state_dict(blob[descriptor['state_key']],strict=True);del blob;gc.collect()
    probe.to(device).eval().requires_grad_(False);compact=compact_records(records);rows=[];started=time.time()
    with torch.inference_mode():
        for offset in range(0,len(views),8):
            part=views[offset:offset+8];data=capture_packed(actor,compact,part,separator,device)
            with torch.autocast('cuda',dtype=torch.bfloat16):values=(torch.sigmoid(probe(data))/4).float().cpu().tolist()
            rows.extend(dict(v,predicted_variance=x)for v,x in zip(part,values))
            old.atomic(progress,dict(stage='FROZEN_FIXEDG5_ONE_QUESTION_PACKED_VARIANCE_PREFILL',rank=rank,questions_scored=len(rows),expected_questions=len(views),actor_rows_per_group=1,question_prompt_occurrences_per_group=1,M=5,prefix=20,layer=21,selection_score='predicted_population_EM_variance',seconds=time.time()-started,actor_updates=0,probe_updates=0,time=time.time()))
            del data
    del actor,probe;gc.collect();torch.cuda.empty_cache();return rows


def paired_summary(rows,repeats=2000,seed=20261022):
    """Different declared BA thresholds; both rankings preserve native scores."""
    import numpy as np
    from .online_packed_variance_scaling import metrics as variance_metrics
    rows=sorted(rows,key=lambda r:r['qid']);n=len(rows)
    if n!=2048 or len({r['qid']for r in rows})!=n or repeats<20:raise ValueError('One full fixed2048 pool and declared bootstrap required')
    y=np.asarray([r['true_mixed']for r in rows],dtype=bool)
    z=np.asarray([r['baseline_logit']for r in rows],dtype=float);p=np.asarray([r['candidate_variance']for r in rows],dtype=float)
    if not np.isfinite(z).all()or not np.isfinite(p).all()or not y.any()or y.all()or np.any((p<0)|(p>.25)):raise ValueError('Finite legal variance and two classes required')
    pred0=z>=0;pred1=p>=THRESHOLD
    def ba(pred,truth):return float(.5*(pred[truth].mean()+(~pred[~truth]).mean()))
    def top(score,truth):return float(truth[np.argsort(-score,kind='stable')[:512]].mean())
    bce=np.maximum(z,0)-z*y+np.log1p(np.exp(-np.abs(z)))
    base=dict(questions=n,binary_Mixed_BCE=float(bce.mean()),balanced_accuracy=ba(pred0,y),Mixed_recall=float(pred0[y].mean()),nonMixed_false_positive_rate=float(pred0[~y].mean()),selected512=dict(questions=512,Mixed=int(top(z,y)*512),Mixed_fraction=top(z,y)),threshold_probability=.5)
    for name,k in(('all_wrong',0),('all_correct',5)):
        mask=np.array([sum(r['labels'])==k for r in rows]);base[name]=dict(questions=int(mask.sum()),Mixed_false_positive_rate=float(pred0[mask].mean())if mask.any()else None)
    candidate=variance_metrics([dict(r,predicted_variance=r['candidate_variance'])for r in rows],True)
    observed=[ba(pred1,y)-ba(pred0,y),top(p,y)-top(z,y)];rng=np.random.default_rng(seed);samples=[]
    for _ in range(repeats):
        idx=rng.integers(0,n,size=n);yy=y[idx]
        if yy.any()and not yy.all():samples.append([ba(pred1[idx],yy)-ba(pred0[idx],yy),top(p[idx],yy)-top(z[idx],yy)])
    ci=np.quantile(np.asarray(samples),[.025,.975],axis=0)
    ids0={rows[i]['qid']for i in np.argsort(-z,kind='stable')[:512]};ids1={rows[i]['qid']for i in np.argsort(-p,kind='stable')[:512]}
    return dict(baseline_E1_EMA=base,candidate_variance=candidate,paired_difference_candidate_minus_E1_EMA=dict(Mixed_balanced_accuracy=observed[0],Top512_true_Mixed_fraction=observed[1],percentile95_question_bootstrap=dict(Mixed_balanced_accuracy=ci[:,0].tolist(),Top512_true_Mixed_fraction=ci[:,1].tolist()),selected512_overlap=len(ids0&ids1)),bootstrap=dict(seed=seed,repeats=len(samples),unit='question',paired=True,reselect_Top512_in_each_resample=True,limitation='within-pool question uncertainty only; not seed/dataset uncertainty'),thresholds=dict(baseline_Mixed_probability=.5,candidate_population_variance=.08),candidate_Mixed_BCE_not_defined=True,threshold_fitted=False,strict_EM_used_for_ranking=False)


def run(freeze_path,freeze_sha):
    if os.uname().sysname!='Linux'or not os.environ.get('SLURM_JOB_ID')or not os.environ.get('CUDA_VISIBLE_DEVICES'):raise RuntimeError('Owned remote2GPU Slurm allocation required')
    freeze_path=old.remote(freeze_path)
    if old.proof(freeze_path)['sha256']!=freeze_sha:raise ValueError('Frozen candidate SHA changed')
    frozen=old.read(freeze_path);freeze_contract(frozen)
    if any((previous/'final-use-started.json').exists()for previous in PREVIOUS_BASES):raise RuntimeError('Previous final scorer already unblinded pools; preserve its frozen protocol')
    if frozen['freeze_epoch']>time.time():raise ValueError('Future freeze timestamp')
    old.checked(frozen['viewed_validation_evaluation']);old.checked(frozen['scorer_publication'])
    pub=old.read(frozen['scorer_publication']['path']);code=Path(__file__).resolve().parents[3]
    if pub['code_root']!=str(code):raise ValueError('Different immutable scorer publication')
    for name,ref in pub['files'].items():old.checked(dict(ref,path=str(code/name)))
    official_root=B/'assets/qwen25-7b-instruct-a09a35458c70';official=old.read(official_root/'complete.json')
    if official['revision']!=REVISION or official['status']!='MIGRATION_ALL_SELECTED_ASSETS_VERIFIED':raise ValueError('Verified frozen official actor required')
    for name,ref in official['files'].items():
        if Path(name).name!=name:raise ValueError('Invalid official filename')
        old.checked(dict(ref,path=str(official_root/name)))
    import torch
    import torch.distributed as dist
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE'])
    if world!=2:raise ValueError('Two ranks required')
    device=torch.device('cuda',int(os.environ['LOCAL_RANK']));torch.cuda.set_device(device);dist.init_process_group('nccl')
    BASE.mkdir(exist_ok=True);guard=(BASE/('scoring-rank-%d.lock'%rank)).open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if(BASE/'final-use-started.json').exists():
        if old.read(BASE/'final-use-started.json')['freeze']!=old.proof(freeze_path):raise ValueError('Final already used by another frozen candidate')
    elif rank==0:old.atomic(BASE/'final-use-started.json',dict(freeze=old.proof(freeze_path),time=time.time(),further_tuning_allowed=False))
    dist.barrier()
    if old.proof(old.MATERIALIZATION)['sha256']!=old.MATERIALIZATION_SHA:raise ValueError('Two original final identities changed')
    materialized=old.read(old.MATERIALIZATION);allq=set();results={};proofs={}
    for pool_name in('confirmation2048','stability2048'):
        views,records,sources=old.load_pool(materialized['pools'][pool_name],rank,world)
        if allq&{r['qid']for r in views}:raise ValueError('Two final pools overlap')
        allq|={r['qid']for r in views}
        for view in views:
            if len(view['trajectory_ids'])!=5:raise ValueError('Fixed ownG5 required')
            mean=sum(view['labels'])/5;view.update(M=5,target_variance=mean*(1-mean),target4variance=4*mean*(1-mean))
        baseline=old.score_variant(dict(frozen['baseline'],name='baseline'),views,records,device,BASE/('progress-rank-%d.json'%rank),rank)
        candidate=score_candidate(frozen['candidate'],views,records,device,BASE/('progress-rank-%d.json'%rank),rank)
        bs={r['qid']:r['logit']for r in baseline};cs={r['qid']:r['predicted_variance']for r in candidate}
        paired=[dict(v,baseline_logit=bs[v['qid']],candidate_variance=cs[v['qid']])for v in views]
        gathered=[None]*world;dist.all_gather_object(gathered,paired)
        if rank==0:
            rows=sorted([v for part in gathered for v in part],key=lambda r:r['qid']);path=BASE/(pool_name+'-paired-predictions.jsonl.gz');tmp=path.with_suffix('.tmp')
            with gzip.open(tmp,'wt')as output:
                for row in rows:output.write(json.dumps(row)+'\n')
            tmp.replace(path);results[pool_name]=paired_summary(rows);results[pool_name]['predictions']=old.proof(path);proofs[pool_name]=sources
            old.atomic(BASE/(pool_name+'-complete.json'),dict(metrics=results[pool_name],freeze=old.proof(freeze_path),sources=sources))
        del records;dist.barrier()
    if rank==0:old.atomic(BASE/'complete.json',dict(status='FROZEN_FINAL4096_PACKED_VARIANCE_PAIRED_E1_EMA_COMPLETE',freeze=old.proof(freeze_path),pools=results,sources=proofs,questions=4096,fixed_G5_trajectories=20480,actor_updates=0,probe_updates=0,hidden_features_persisted=False,further_tuning_on_final_allowed=False,threshold_fitted=False,final_metrics_used_for_candidate_selection=False,time=time.time()))
    dist.destroy_process_group()


def main():
    p=argparse.ArgumentParser();p.add_argument('--freeze-json',required=True);p.add_argument('--freeze-sha',required=True);args=p.parse_args()
    try:run(args.freeze_json,args.freeze_sha)
    except BaseException as error:
        if os.uname().sysname=='Linux'and BASE.exists():old.atomic(BASE/('failure-rank-%s.json'%os.environ.get('RANK','unknown')),dict(error=repr(error),traceback=traceback.format_exc(),time=time.time()))
        raise

if __name__=='__main__':main()
