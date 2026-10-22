"""Read-only GPU audit of a published packed Z E8 checkpoint on viewed dev2048.

Only the order of the same five genuine trajectories changes. No final-pool
access, optimizer, training state mutation, new rollout or new target sampling.
"""
import argparse, copy, fcntl, gc, gzip, hashlib, inspect, json, os, time, traceback
from pathlib import Path

B = Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019')
A = B/'algorithm-improvement-20261021'
BASE = A/'packed-Z-E8-trajectory-permutation-audit-20261023-v1'
ORDERS = ('original', 'reverse', 'cyclic1')
SEPARATOR = [198, 14374, 34682, 198]

def canonical(v): return json.dumps(v, sort_keys=True, separators=(',', ':')).encode()
def digest(v): return hashlib.sha256(canonical(v)).hexdigest()
def read(p): return json.loads(Path(p).read_bytes())
def atomic(p, v):
    p=Path(p); tmp=p.with_name(p.name+'.%d.tmp'%os.getpid()); tmp.write_bytes(canonical(v)+b'\n'); tmp.replace(p)
def proof(p):
    p=Path(p); h=hashlib.sha256()
    with p.open('rb') as stream:
        for data in iter(lambda:stream.read(8*1024**2), b''): h.update(data)
    return dict(path=str(p), bytes=p.stat().st_size, sha256=h.hexdigest())
def checked(ref):
    p=Path(ref['path']); assert proof(p)=={k:ref[k] for k in ('path','bytes','sha256')}, str(p)
    return p

def reorder(view, order):
    """A permutation of complete (own trajectory ID, strict EM) pairs only."""
    assert order in ORDERS and len(view['trajectory_ids'])==len(view['labels'])==5
    assert len(set(view['trajectory_ids']))==5 and all(y in (0,1) for y in view['labels'])
    indices=list(range(5)) if order=='original' else list(reversed(range(5))) if order=='reverse' else [1,2,3,4,0]
    out=dict(view, trajectory_ids=[view['trajectory_ids'][i] for i in indices], labels=[view['labels'][i] for i in indices])
    mean=sum(out['labels'])/5; out.update(M=5, target_variance=mean*(1-mean), target4variance=4*mean*(1-mean), true_mixed=0<sum(out['labels'])<5)
    return out

def compact(row):
    policy=[]
    for segment in row['segments']:
        take=min(20-len(policy),segment['policy_length'])
        policy+=segment['context_tokens'][segment['request_length']:segment['request_length']+take]
        if len(policy)==20: break
    assert row['prompt'] and 0<len(policy)<=20
    return dict(qid=row['qid'],trajectory_id=row['trajectory_id'],prompt=row['prompt'],policy20=policy)

def input_identity(view, records, pack):
    rows=[records[t] for t in view['trajectory_ids']]
    assert all(r['qid']==view['qid'] for r in rows)
    packed=pack(view,records,SEPARATOR)
    return dict(qid=view['qid'], unordered_own_ID_EM_tokens=digest(sorted((tid,y,records[tid]['prompt'],records[tid]['policy20']) for tid,y in zip(view['trajectory_ids'],view['labels']))),
        ordered_ID_EM=digest(list(zip(view['trajectory_ids'],view['labels']))),packed_input_sha256=digest(packed),target_variance=view['target_variance'])

def ranks(values):
    import numpy as np
    order=np.argsort(values,kind='stable'); result=np.empty(len(values),dtype=float); i=0
    while i<len(order):
        j=i+1
        while j<len(order) and values[order[j]]==values[order[i]]: j+=1
        result[order[i:j]]=(i+j-1)/2; i=j
    return result

def corr(a,b):
    import numpy as np
    return float(np.corrcoef(a,b)[0,1]) if np.std(a)>0 and np.std(b)>0 else None

def summary(rows, key):
    import numpy as np
    scores=np.asarray([r[key] for r in rows]); target=np.asarray([r['target_variance'] for r in rows]); labels=np.asarray([r['true_mixed'] for r in rows],dtype=bool); pred=scores>=.08
    assert len(rows)==2048 and len({r['qid'] for r in rows})==2048
    tp=int((pred&labels).sum()); tn=int((~pred&~labels).sum()); pos=int(labels.sum()); neg=len(rows)-pos
    top=sorted(rows,key=lambda r:(-r[key],r['qid']))[:512]; classes={}
    for name,mask in [('all_wrong',np.asarray([sum(r['labels'])==0 for r in rows])),('Mixed',labels),('all_correct',np.asarray([sum(r['labels'])==5 for r in rows]))]:
        classes[name]=dict(questions=int(mask.sum()),predicted_Mixed=int(pred[mask].sum()),predicted_nonMixed=int((~pred[mask]).sum()),mean_predicted_variance=float(scores[mask].mean()))
    return dict(variance_MSE=float(((scores-target)**2).mean()),variance_MAE=float(np.abs(scores-target).mean()),variance_Spearman=corr(ranks(scores),ranks(target)),variance_Pearson=corr(scores,target),
        BA=float(.5*(tp/pos+tn/neg)),variance_threshold=.08,threshold_fitted=False,confusion=dict(tp=tp,tn=tn,fp=neg-tn,fn=pos-tp),classwise=classes,
        Top512=dict(Mixed=sum(r['true_mixed'] for r in top),Mixed_fraction=sum(r['true_mixed'] for r in top)/512,selected_qids_sha256=digest([r['qid'] for r in top])))

def compare(rows, state, order):
    import numpy as np
    original=state+'_original'; current=state+'_'+order
    x=np.asarray([r[original] for r in rows]); y=np.asarray([r[current] for r in rows]); delta=np.abs(y-x)
    top=lambda key:{r['qid'] for r in sorted(rows,key=lambda r:(-r[key],r['qid']))[:512]}
    ld=np.abs(np.asarray([r[current+'_logit']-r[original+'_logit'] for r in rows]))
    return dict(absolute_variance_delta_mean=float(delta.mean()),absolute_variance_delta_max=float(delta.max()),absolute_variance_delta_p95=float(np.quantile(delta,.95)),
        absolute_logit_delta_mean=float(ld.mean()),absolute_logit_delta_max=float(ld.max()),score_Spearman=corr(ranks(x),ranks(y)),Top512_overlap=len(top(original)&top(current)),
        classification_changed_questions=int(((x>=.08)!=(y>=.08)).sum()))

def run(config_path, config_sha):
    assert os.uname().sysname=='Linux' and os.environ.get('SLURM_JOB_ID')
    assert proof(config_path)['sha256']==config_sha
    cfg=read(config_path); assert cfg['status']=='VIEWED_DEV_E8_PACKED_PERMUTATION_AUDIT_BOUND'
    assert cfg['base']==str(BASE) and cfg['checkpoint']['sha256']=='2d81f15cd4af00fe192963cba6f6db86aba22bf47523bd3e10982b86f7178483'
    pub=read(checked(cfg['source_publication'])); code=Path(__file__).resolve().parents[3]; assert pub['code_root']==str(code)
    for name, ref in pub['files'].items(): checked(dict(ref,path=str(code/name)))
    official=read(checked(cfg['official_inventory']))
    assert official['revision']=='a09a35458c702b33eeacc393d103063234e8bc28' and official['status']=='MIGRATION_ALL_SELECTED_ASSETS_VERIFIED'
    checked(cfg['source_training_publication'])
    import numpy as np
    import torch
    import torch.distributed as dist
    from torch import nn
    from transformers import AutoModel, AutoTokenizer
    from .packed_variance_probe import PackedVarianceHead, PackedVarianceProbe, pack_group
    rank=int(os.environ['RANK']); world=int(os.environ['WORLD_SIZE']); assert world==2
    device=torch.device('cuda',int(os.environ['LOCAL_RANK'])); torch.cuda.set_device(device); dist.init_process_group('nccl')
    guard=(BASE/('audit-rank-%d.lock'%rank)).open('a+'); fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (BASE/'complete.json').exists(): raise RuntimeError('This independent audit is already complete')
    if rank==0:
        checked(cfg['checkpoint']); checked(cfg['source_evaluation']); checked(cfg['source_saved_predictions'])
    dist.barrier()
    groups=read(checked(cfg['fixed_validation'])); assert len(groups)==2048
    allviews=[reorder(groups[q],'original') for q in sorted(groups)]; mine=allviews[rank::world]
    assert sum(v['true_mixed'] for v in allviews)==663
    wanted={t for v in mine for t in v['trajectory_ids']}; records={}; fixture=read(checked(cfg['fixture']))
    for index,ref in enumerate(fixture['token_shards']):
        if index%world!=rank: continue
        with gzip.open(checked(ref),'rt') as stream:
            for line in stream:
                row=json.loads(line)
                if row['trajectory_id'] in wanted: records[row['trajectory_id']]=compact(row)
    assert set(records)==wanted
    if rank==0: atomic(BASE/'controller.json',dict(job_id=os.environ['SLURM_JOB_ID'],world_size=2,checkpoint=cfg['checkpoint'],epochs=8,cursor=10568,
        fixed_questions=2048,M=5,orders=list(ORDERS),layer=21,prefix=20,question_prompt_occurrences_per_group=1,actor_rows_per_group=1,
        raw_and_EMA=True,selection_score='predicted_population_EM_variance',classification_variance_threshold=.08,optimizer_created=False,actor_updates=0,probe_updates=0,final4096_read=False,time=time.time()))
    torch.manual_seed(9137); np.random.seed(9137)
    model_dir=B/'assets/qwen25-7b-instruct-a09a35458c70'; tokenizer=AutoTokenizer.from_pretrained(str(model_dir),local_files_only=True); assert tokenizer.encode('\n### trajectory\n',add_special_tokens=False)==SEPARATOR
    actor=AutoModel.from_pretrained(str(model_dir),local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='flash_attention_2').to(device).eval().requires_grad_(False)
    block=copy.deepcopy(actor.layers[21]).float(); rotary=copy.deepcopy(actor.rotary_emb).float(); actor.layers=nn.ModuleList(list(actor.layers[:21])); actor.norm=nn.Identity(); torch.cuda.empty_cache()
    raw=PackedVarianceProbe(block,rotary,'cache_position' in inspect.signature(block.forward).parameters,PackedVarianceHead()); ema=copy.deepcopy(raw)
    blob=torch.load(cfg['checkpoint']['path'],map_location='cpu',mmap=True,weights_only=False)
    assert (blob['architecture'],blob['epoch'],blob['cursor'],blob['layer'],blob['prefix'],blob['loss_name'],blob['EMA_decay'])==('one_question_M3to10_policy20_packed_variance_v1',8,10568,21,20,'z_mse_anchor',.9995)
    raw.load_state_dict(blob['raw'],strict=True); ema.load_state_dict(blob['EMA'],strict=True); del blob; gc.collect()
    raw.to(device).eval().requires_grad_(False); ema.to(device).eval().requires_grad_(False)
    rows={v['qid']:dict(v) for v in mine}; identities={}; started=time.time()
    with torch.inference_mode():
        for order in ORDERS:
            views=[reorder(v,order) for v in mine]; ids=[input_identity(v,records,pack_group) for v in views]; identities[order]=ids
            for offset in range(0,len(views),8):
                part=views[offset:offset+8]; packed=[pack_group(v,records,SEPARATOR) for v in part]; width=max(len(v['input_ids']) for v in packed)
                tokens=torch.full((len(part),width),151643,dtype=torch.long,device=device); mask=torch.zeros_like(tokens,dtype=torch.bool); positions=torch.zeros_like(tokens); types=torch.zeros_like(tokens)
                for j,item in enumerate(packed):
                    n=len(item['input_ids']); tokens[j,:n]=torch.tensor(item['input_ids'],device=device); mask[j,:n]=torch.tensor(item['attention_mask'],device=device); positions[j,:n]=torch.tensor(item['position_ids'],device=device); types[j,:n]=torch.tensor(item['token_types'],device=device)
                features=actor(input_ids=tokens,attention_mask=mask,position_ids=positions,use_cache=False,return_dict=True).last_hidden_state.detach(); data=dict(features=features,attention_mask=mask,position_ids=positions,token_types=types)
                with torch.autocast('cuda',dtype=torch.bfloat16): logits={'raw':raw(data),'EMA':ema(data)}
                for state,z in logits.items():
                    zs=z.float().cpu().tolist(); vs=(torch.sigmoid(z)/4).float().cpu().tolist()
                    for view,logit,value in zip(part,zs,vs): rows[view['qid']][state+'_'+order]=value; rows[view['qid']][state+'_'+order+'_logit']=logit
                atomic(BASE/('progress-rank-%d.json'%rank),dict(stage='VIEWED_DEV_TRAJECTORY_ORDER_AUDIT',order=order,questions_scored=offset+len(part),expected_own_questions=len(views),rank=rank,seconds=time.time()-started,time=time.time()))
                del features,data,logits
    for order in ORDERS[1:]:
        assert [x['unordered_own_ID_EM_tokens'] for x in identities[order]]==[x['unordered_own_ID_EM_tokens'] for x in identities['original']]
        assert [x['target_variance'] for x in identities[order]]==[x['target_variance'] for x in identities['original']]
    received=[None]*world; dist.all_gather_object(received,dict(rows=list(rows.values()),identities=identities))
    if rank==0:
        merged=sorted([r for part in received for r in part['rows']],key=lambda r:r['qid']); assert len(merged)==2048
        saved={}
        with gzip.open(cfg['source_saved_predictions']['path'],'rt') as stream:
            for line in stream:
                value=json.loads(line); saved[value['qid']]=value
        assert set(saved)=={r['qid'] for r in merged}
        for r in merged: assert (r['trajectory_ids'],r['labels'])==(saved[r['qid']]['trajectory_ids'],saved[r['qid']]['labels'])
        ident={order:sorted([x for part in received for x in part['identities'][order]],key=lambda r:r['qid']) for order in ORDERS}
        path=BASE/'dev2048-permutation-predictions.jsonl.gz'
        with gzip.open(path,'wt') as stream:
            for row in merged: stream.write(json.dumps(row)+'\n')
        results={state:{order:summary(merged,state+'_'+order) for order in ORDERS} for state in ('raw','EMA')}
        comparisons={state:{order:compare(merged,state,order) for order in ORDERS[1:]} for state in ('raw','EMA')}
        saved_delta={state:dict(mean_absolute_variance_delta=float(np.mean([abs(r[state+'_original']-saved[r['qid']][state+'_predicted_variance']) for r in merged])),max_absolute_variance_delta=float(max(abs(r[state+'_original']-saved[r['qid']][state+'_predicted_variance']) for r in merged))) for state in ('raw','EMA')}
        atomic(BASE/'complete.json',dict(status='VIEWED_DEV2048_PACKED_E8_TRAJECTORY_PERMUTATION_AUDIT_COMPLETE',checkpoint=cfg['checkpoint'],config=proof(config_path),source_evaluation=cfg['source_evaluation'],
            metrics=results,comparisons_to_original=comparisons,original_vs_training_E8_saved_predictions=saved_delta,identity={order:digest(ident[order]) for order in ORDERS},same_own_ID_EM_prefix_tokens_and_targets_all_orders=True,
            predictions=proof(path),questions=2048,states=['raw','EMA'],orders=list(ORDERS),fixed_M=5,selection_score='predicted_population_EM_variance',variance_threshold=.08,threshold_fitted=False,
            interpretation='This measures order sensitivity on an already viewed fixed dev pool; not final confirmation or new-question generalization.',actor_updates=0,probe_updates=0,optimizer_created=False,final4096_read=False,seconds=time.time()-started,time=time.time()))
    dist.barrier(); dist.destroy_process_group()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--config-sha',required=True); args=p.parse_args()
    try: run(args.config,args.config_sha)
    except BaseException:
        if BASE.is_dir(): atomic(BASE/('failure-rank-%s.json'%os.environ.get('RANK','unknown')),dict(time=time.time(),traceback=traceback.format_exc()))
        raise
if __name__=='__main__': main()
