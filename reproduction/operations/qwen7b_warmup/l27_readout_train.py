"""Independent L27 readout experiment: E1 block/Adam + newly seeded head.

This is NOT whole-model lossless resume. Raw/EMA block tensors and block Adam
are inherited; the old flattened readout/Adam are deliberately NOT inherited.
New readout (and optional attention key) starts Adam0. Two explicit optimizers
retain independently checked10464+cursor and cursor counters. No job submission.
"""
import argparse
import copy
import fcntl
import os
from pathlib import Path
import signal
import time
import traceback
from contextlib import nullcontext
import torch
from . import probe_ablation_train as u
from .probe_ablation_loss import probe_ablation_loss
from .probe_ablation_model import TokenReadout,ProbeAblationSpec,NativeBlockProbe,CAPTURE_KIND

STOP=False
EVENT_LOSS=dict(trajectory_weight=1.,three_event_weight=1.)


def parameter_partition(model):
    block=dict(model.block.named_parameters())
    head={name:p for name,p in model.named_parameters() if not name.startswith('block.')}
    if not block or not head or any(not(name.startswith('readout.') or name.startswith('output.')) for name in head):
        raise ValueError('Only the native block and explicit new readout may be trained')
    return block,head


def reset_head(model,seed):
    if type(seed) is not int:raise ValueError('Explicit common head seed required')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if hasattr(model,'readout'):
            model.readout.output.reset_parameters()
            if model.readout.attention_key is not None:
                with torch.no_grad():model.readout.attention_key.zero_()
        else:model.output.reset_parameters()


def adam_count(optimizer):
    if not optimizer.state:return 0
    parameters=[p for group in optimizer.param_groups for p in group['params']]
    if set(optimizer.state)!=set(parameters):raise ValueError('Partially populated optimizer states')
    counts={int(state['step']) for state in optimizer.state.values()}
    if len(counts)!=1:raise ValueError('Adam steps differ within one parameter role')
    return counts.pop()


def move_adam(optimizer,device):
    for state in optimizer.state.values():
        for name,value in state.items():
            if torch.is_tensor(value) and name!='step':state[name]=value.to(device)


def initialize_branch(source,factory,head_seed,device):
    """Injectable factory for small CPU tests; production factories are fixed."""
    model=factory();reset_head(model,head_seed);ema=copy.deepcopy(model).eval().requires_grad_(False)
    block,head=parameter_partition(model)
    for own,field in ((model,'raw'),(ema,'ema')):
        selected={name[len('block.'):]:value for name,value in source[field].items() if name.startswith('block.')}
        own.block.load_state_dict(selected,strict=True)
        if u.model_mapping_hash(own.block.state_dict())!=u.model_mapping_hash(selected):
            raise ValueError('Copied E1 '+field+' block is not tensor-exact')
    u.restore_buffers(model,source['raw_buffers']);u.restore_buffers(ema,source['ema_buffers'])
    model=model.to(device);ema=ema.to(device);block,head=parameter_partition(model)
    old=source['optimizer']
    if len(old['param_groups'])!=1:raise ValueError('Pinned E1 legacy Adam must have one parameter group')
    group=old['param_groups'][0];ids=group['params']
    expected={'block.'+name for name in block}|{'output.weight','output.bias'}
    names=[name for name in source['raw'] if name in expected]
    if set(names)!=expected or len(names)!=len(ids):raise ValueError('Legacy named-parameter/Adam binding differs')
    old_ids=dict(zip(names,ids));block_states={}
    for new_id,(name,parameter) in enumerate(block.items()):
        state=copy.deepcopy(old['state'][old_ids['block.'+name]])
        if int(state['step'])!=source['steps'] or any(state[k].shape!=parameter.shape for k in ('exp_avg','exp_avg_sq')):
            raise ValueError('Inherited block Adam step/geometry differs for '+name)
        block_states[new_id]=state
    block_optimizer=torch.optim.Adam(block.values(),lr=group['lr'])
    block_group=copy.deepcopy(group);block_group['params']=list(range(len(block)))
    block_optimizer.load_state_dict(dict(state=block_states,param_groups=[block_group]))
    head_optimizer=torch.optim.Adam(head.values(),lr=group['lr'])
    head_group=copy.deepcopy(group);head_group['params']=list(range(len(head)))
    head_optimizer.load_state_dict(dict(state={},param_groups=[head_group]))
    move_adam(block_optimizer,device);move_adam(head_optimizer,device)
    canonical=[s for s in source['rank_states'] if s['rank']==0]
    if len(canonical)!=1:raise ValueError('Exactly one source canonical rank0 RNG required')
    u.restore_rng(canonical[0]['rng'],device)
    if adam_count(block_optimizer)!=source['steps'] or adam_count(head_optimizer)!=0:
        raise ValueError('Block/head independent initialization counters invalid')
    proof=dict(source_block_steps=source['steps'],new_head_steps=0,head_seed=head_seed,
               block_raw_inherited_exact=True,block_ema_inherited_exact=True,
               block_adam_inherited=True,new_head_adam_reset=True,
               source_old_head_inherited=False,whole_model_lossless_resume=False,
               initial_raw_block_sha256=u.model_mapping_hash(model.block.state_dict()),
               initial_ema_block_sha256=u.model_mapping_hash(ema.block.state_dict()),
               new_head_parameter_names=list(head),new_head_parameter_sha256=u.tensor_hash(head),
               canonical_source_rng_rank=0,source_world_size=4,current_world_size=1)
    return model,ema,block_optimizer,head_optimizer,proof


def branch_hashes(model,ema,block_optimizer,head_optimizer,device):
    return dict(raw=u.model_hash(model),ema=u.model_hash(ema),
                block_adam=u.optimizer_hash(block_optimizer),head_adam=u.optimizer_hash(head_optimizer),
                raw_buffers=u.tensor_hash(u.buffers(model)),ema_buffers=u.tensor_hash(u.buffers(ema)),
                rng=u.rng_hash(u.rng_state(device)))


def restore_branch(saved,model,ema,block_optimizer,head_optimizer,device):
    model.load_state_dict(saved['raw'],strict=True);ema.load_state_dict(saved['ema'],strict=True)
    u.restore_buffers(model,saved['raw_buffers']);u.restore_buffers(ema,saved['ema_buffers'])
    for optimizer,key in ((block_optimizer,'block_optimizer'),(head_optimizer,'head_optimizer')):
        optimizer.load_state_dict(copy.deepcopy(saved[key]));move_adam(optimizer,device)
    u.restore_rng(saved['rng'],device)
    if (adam_count(block_optimizer)!=saved['block_steps'] or adam_count(head_optimizer)!=saved['head_steps']):
        raise ValueError('Restored independent block/head counters differ')
    if branch_hashes(model,ema,block_optimizer,head_optimizer,device)!=saved['accepted_hashes']:
        raise ValueError('Restored branch raw/EMA/Adam/buffer/RNG hash mismatch')


def update(model,ema,block_optimizer,head_optimizer,reader,lookup,qids,config,device):
    before=(adam_count(block_optimizer),adam_count(head_optimizer));model.train()
    block_optimizer.zero_grad(set_to_none=True);head_optimizer.zero_grad(set_to_none=True)
    selected=[lookup[q] for q in qids];parts=[];started=time.monotonic()
    for offset in range(0,len(selected),config['micro_groups']):
        own=selected[offset:offset+config['micro_groups']]
        data=reader.get([tid for g in own for tid in g['trajectory_ids']],device)
        context=torch.autocast('cuda',dtype=torch.bfloat16) if device.type=='cuda' else nullcontext()
        with context:parts.append(model(data).reshape(len(own),5))
    logits=torch.cat(parts);labels=torch.tensor([g['labels'] for g in selected],device=device,dtype=logits.dtype)
    result=probe_ablation_loss(logits,labels,**EVENT_LOSS)
    if not bool(torch.isfinite(result['loss'])):raise ValueError('Nonfinite readout loss')
    result['loss'].backward()
    if not all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
        raise ValueError('Missing/nonfinite readout or block gradient')
    block_optimizer.step();head_optimizer.step()
    if (adam_count(block_optimizer),adam_count(head_optimizer))!=(before[0]+1,before[1]+1):
        raise ValueError('Both parameter roles must advance exactly once')
    with torch.no_grad():
        for target,value in zip(ema.parameters(),model.parameters()):target.mul_(.99).add_(value,alpha=.01)
    return dict(loss=float(result['loss'].detach()),trajectory_bce=float(result['trajectory_bce'].detach()),
                group_event_nll=float(result['group_event_nll'].detach()),
                block_steps=adam_count(block_optimizer),head_steps=adam_count(head_optimizer),
                seconds=time.monotonic()-started,actor_updates=0)


def save(base,model,ema,block_optimizer,head_optimizer,device,cursor,binding,source_ranks):
    root=base/'checkpoints';root.mkdir(exist_ok=True);path=root/('cursor-%06d.pt'%cursor)
    if path.exists():raise ValueError('Do not overwrite an immutable readout checkpoint')
    blob=dict(role='E1_L27_BLOCK_ADAM_INHERITED_NEW_READOUT_BRANCH',cursor=cursor,
              raw=model.state_dict(),ema=ema.state_dict(),raw_buffers=u.buffers(model),ema_buffers=u.buffers(ema),
              block_optimizer=block_optimizer.state_dict(),head_optimizer=head_optimizer.state_dict(),
              block_steps=adam_count(block_optimizer),head_steps=adam_count(head_optimizer),
              rng=u.rng_state(device),source_all_rank_states=source_ranks,
              accepted_hashes=branch_hashes(model,ema,block_optimizer,head_optimizer,device),
              whole_model_lossless_resume=False,within_branch_full_state_resume_supported=True,
              actor_updates=0,**binding)
    temporary=path.with_suffix('.tmp')
    with temporary.open('wb') as f:torch.save(blob,f);f.flush();os.fsync(f.fileno())
    temporary.replace(path);check=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
    for key in ('raw','ema'):
        if not all(bool(torch.isfinite(v).all()) for v in check[key].values()):raise ValueError('Nonfinite saved parameters')
    for key,expected in (('block_optimizer',check['block_steps']),('head_optimizer',check['head_steps'])):
        for state in check[key]['state'].values():
            if int(state['step'])!=expected or any(not bool(torch.isfinite(state[k]).all()) for k in ('exp_avg','exp_avg_sq')):
                raise ValueError('Invalid saved '+key)
    actual=dict(raw=u.model_mapping_hash(check['raw']),ema=u.model_mapping_hash(check['ema']),
                block_adam=u.optimizer_blob_hash(check['block_optimizer']),head_adam=u.optimizer_blob_hash(check['head_optimizer']),
                raw_buffers=u.tensor_hash(check['raw_buffers']),ema_buffers=u.tensor_hash(check['ema_buffers']),rng=u.rng_hash(check['rng']))
    if actual!=check['accepted_hashes']:raise ValueError('CPU saved-state component hash mismatch')
    receipt=dict(path=str(path),sha256=u.sha(path),bytes=path.stat().st_size,cursor=cursor,
                 block_steps=check['block_steps'],head_steps=check['head_steps'],
                 accepted_hashes=actual,config_sha256=binding['config_sha256'])
    u.atomic(base/'latest.json',receipt);return receipt


class VerifiedLayer26Reader:
    def __init__(self,manifests,reader_factory=None):
        if reader_factory is None:
            from .packed_features import FeatureReader
            reader_factory=FeatureReader
        self.manifests=[]
        for path in manifests:
            item=u.read(path)
            if (item['status']!='LOSSLESS_NATIVE_FEATURES_COMPLETE' or item['feature_layer']!=26 or
                    item['dimension']!=3584 or item['question_max_tokens']!=273 or item['prefix_tokens']!=20):
                raise ValueError('Only existing authentic native L26/Prefix20 bank is admitted')
            self.manifests.append(dict(path=str(path),sha256=u.sha(path)))
        if not self.manifests:raise ValueError('No verified native L26 feature manifests')
        self.reader=reader_factory(manifests)

    def get(self,ids,device):
        data=self.reader.get(ids,device)
        return dict(data,capture_layer=26,capture_kind=CAPTURE_KIND)

    def close(self):self.reader.close()

    def alias_role(self,ids):
        if len(ids)!=len(set(ids)):raise ValueError('Alias audit requires distinct declared trajectory IDs')
        lengths=[self.reader.records[tid][1]['policy_length'] for tid in ids]
        return dict(trajectories=len(lengths),empty_prefix=sum(v==0 for v in lengths),
                    minimum_prefix=min(lengths),maximum_prefix=max(lengths))


def last_scope_alias_proof(reader,held_reader,fit,diagnostic,held):
    roles={name:own.alias_role([tid for g in groups for tid in g['trajectory_ids']])
           for name,own,groups in [('fit',reader,fit),('diagnostic512',reader,diagnostic),('selection2048',held_reader,held)]}
    valid=all(item['empty_prefix']==0 for item in roles.values())
    return dict(status='MATHEMATICAL_LAST_TOKEN_SCOPE_ALIAS_VERIFIED' if valid else 'LAST_TOKEN_SCOPES_NOT_ALIASED_EMPTY_PREFIX_EXISTS',
                eligible=valid,roles=roles,scope_pair=['prefix_only','question_prefix'],
                reason='Both select the same final real prefix token iff each declared trajectory has nonempty prefix',
                requires_same_block_head_seed_loss_batches_and_RNG=True,
                real_duplicate_GPU_execution_claimed=False)


def factory_for(config):
    kind=config['readout']['kind'];scope=config['readout']['scope']
    if kind=='flatten':
        if scope!='question_prefix':raise ValueError('Flatten control uses complete legacy question+Prefix20')
        from .model import Probe
        return lambda:Probe(config['model_dir'],273,20)
    spec=ProbeAblationSpec(27,26,kind,scope)
    return lambda:NativeBlockProbe(config['model_dir'],spec)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--alias-check-only',action='store_true');args=parser.parse_args()
    base=args.base.resolve()
    if not str(base).startswith('/data/user/jxu957/experiments/'):
        raise ValueError('Use independent remote shared-storage output')
    base.mkdir(parents=True,exist_ok=True);lock=(base/'readout.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:run(base,args.alias_check_only)
    except BaseException as error:
        u.atomic(base/'failure.json',dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),epoch_time=time.time()));raise
    finally:lock.close()


def run(base,alias_only=False):
    global STOP
    c=u.read(base/'config.json');config_sha=u.sha(base/'config.json')
    if c['recipe']['loss_kwargs']!=EVENT_LOSS or (c['update_budget'],c['global_groups'],c['micro_groups'])!=(128,51,2):
        raise ValueError('Readout comparison fixes event loss,128 updates,B51,micro2')
    if type(c['save_every']) is not int or c['save_every']<1:raise ValueError('Positive save cadence required')
    if c['readout']['block_layer']!=27 or c['readout']['capture_layer']!=26:
        raise ValueError('This comparison needs only nativeL26 -> probeL27')
    required=('source_checkpoint','source_config','source_bank','model_config','model_inventory',
              'bank','features','selection_bank','selection_features','plan')
    paths={name:u.checked(c[name]) for name in required}
    if (c['source_checkpoint']['sha256']!=u.SOURCE_SHA or
            paths['model_config'].resolve()!=(Path(c['model_dir'])/'config.json').resolve() or
            paths['model_inventory'].resolve()!=(Path(c['model_dir'])/'official-inventory.json').resolve()):
        raise ValueError('Bind sealed E1 and verified official model-view metadata')
    repo=Path(__file__).resolve().parents[3]
    required_code={'reproduction/operations/qwen7b_warmup/'+name for name in
                   ('l27_readout_train.py','probe_ablation_train.py','probe_ablation_loss.py',
                    'probe_ablation_model.py','model.py','packed_features.py','native_capture.py','download_model.py','runtime.sh')}
    if not required_code<=set(c['code_hashes']):raise ValueError('Missing runtime source bindings')
    for name,value in c['code_hashes'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts or u.sha(repo/name)!=value:raise ValueError('Runtime code SHA mismatch '+name)
    os.environ['QWEN7B_QUESTION_MAX_TOKENS']='273';os.environ['SEARCH_G1_B_PREFIX_TOKENS']='20'
    bank=u.read(paths['bank']);held=u.read(paths['selection_bank'])['selection_validation'];diagnostic=bank['diagnostic'];plan=u.read(paths['plan'])
    if len(diagnostic)!=512 or len(held)!=2048 or len(plan['cohort_qids'])!=6528:raise ValueError('Fixed6528/512/2048 question budgets required')
    external={g['qid'] for g in diagnostic+held}
    if len(external)!=2560 or {g['qid'] for g in bank['train']}&external:raise ValueError('Training/evaluation overlap')
    lookup=u.validate_plan(plan,bank['train'],128,51,external)
    reader=VerifiedLayer26Reader(u.read(paths['features']));held_reader=VerifiedLayer26Reader(u.read(paths['selection_features']))
    try:
        alias=last_scope_alias_proof(reader,held_reader,[lookup[q] for q in plan['cohort_qids']],diagnostic,held)
        u.atomic(base/'last-scope-alias-check.json',alias)
        if alias_only:print(u.json.dumps(alias));return
        if int(os.environ.get('WORLD_SIZE','1'))!=1 or torch.cuda.device_count()!=1:raise ValueError('One dedicated GPU required')
        device=torch.device('cuda',0);properties=torch.cuda.get_device_properties(device)
        if 'H100' not in properties.name or properties.total_memory<75*1024**3:raise ValueError('>=75GiB H100 required')
        if os.environ.get('FLASH_ATTENTION_DETERMINISTIC')!='1' or os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise ValueError('Deterministic FA/CUBLAS flags required')
        torch.cuda.set_device(device);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
        source=torch.load(paths['source_checkpoint'],map_location='cpu',mmap=True,weights_only=False)
        if (source['steps']!=10464 or source['epoch']!=1 or not source['completed_epoch'] or source['cursor']!=0 or source['actor_updates']!=0
                or source['config_sha256']!=c['source_config']['sha256'] or source['bank_sha256']!=c['source_bank']['sha256']):raise ValueError('E1 source lineage/counters differ')
        if {s['rank'] for s in source['rank_states']}!={0,1,2,3}:raise ValueError('Source four-rank RNG archive missing')
        model,ema,bo,ho,initialization=initialize_branch(source,factory_for(c),c['readout']['head_seed'],device)
        source_ranks=copy.deepcopy(source['rank_states'])
        binding=dict(config_sha256=config_sha,source_checkpoint=c['source_checkpoint'],
                     source_assets={name:c[name] for name in required},readout=c['readout'],initialization=initialization,
                     code_hashes=c['code_hashes'],matched_binding_sha256=u.digest(dict(source_assets={k:c[k]['sha256'] for k in required},
                         head_seed=c['readout']['head_seed'],budget=128,global_groups=51,micro_groups=2,loss=EVENT_LOSS,
                         model_dir=str(Path(c['model_dir']).resolve()),code_hashes=c['code_hashes'])))
        u.atomic(base/'controller.json',dict(status='INDEPENDENT_L27_READOUT_RUNNER_LOCKED',pid=os.getpid(),
                  start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
                  job_id=os.environ.get('SLURM_JOB_ID'),device=properties.name,readout=c['readout'],
                  config_sha256=config_sha,source_checkpoint=c['source_checkpoint'],
                  whole_model_lossless_resume=False,epoch_time=time.time()))
        for sig in (signal.SIGTERM,signal.SIGUSR1,signal.SIGINT):signal.signal(sig,request_stop)
        cursor=0
        if (base/'latest.json').exists():
            receipt=u.read(base/'latest.json');saved=torch.load(u.checked(receipt),map_location='cpu',mmap=True,weights_only=False)
            if saved['config_sha256']!=config_sha or saved['matched_binding_sha256']!=binding['matched_binding_sha256']:raise ValueError('Resume bindings changed')
            cursor=saved['cursor']
            if not 0<=cursor<=128 or (saved['block_steps'],saved['head_steps'])!=(10464+cursor,cursor):raise ValueError('Independent counters/cursor inconsistent')
            for name,key in [('progress.json','cursor'),('update-intent.json','next_cursor')]:
                if (base/name).exists() and u.read(base/name)[key]>cursor:raise ValueError('Unpersisted later transaction: refuse rollback')
            restore_branch(saved,model,ema,bo,ho,device);source_ranks=saved['source_all_rank_states'];del saved
        else:
            if (base/'progress.json').exists():raise ValueError('Existing progress without checkpoint')
            save(base,model,ema,bo,ho,device,0,binding,source_ranks)
        u.atomic(base/'restore-acceptance.json',dict(cursor=cursor,block_steps=adam_count(bo),head_steps=adam_count(ho),
                  whole_model_lossless_resume=False,within_branch_full_state_resume_supported=True,
                  accepted_hashes=branch_hashes(model,ema,bo,ho,device),epoch_time=time.time()))
        # Every new process qualifies its real next update; keeps second exact
        # execution as ONE accepted update, with separate block/head counts.
        if cursor<128 and not STOP:
            receipt=u.read(base/'latest.json');start=torch.load(u.checked(receipt),map_location='cpu',mmap=True,weights_only=False)
            update(model,ema,bo,ho,reader,lookup,plan['batches'][cursor],c,device);expected=branch_hashes(model,ema,bo,ho,device)
            restore_branch(start,model,ema,bo,ho,device)
            metrics=update(model,ema,bo,ho,reader,lookup,plan['batches'][cursor],c,device);actual=branch_hashes(model,ema,bo,ho,device)
            u.atomic(base/('next-update-replay-cursor%06d.json'%cursor),dict(expected=expected,actual=actual,passed=actual==expected,
                         source_cursor=cursor,block_steps=adam_count(bo),head_steps=adam_count(ho),whole_model_lossless_resume=False))
            if actual!=expected:raise ValueError('Real readout next-update replay mismatch')
            cursor+=1;save(base,model,ema,bo,ho,device,cursor,binding,source_ranks);write_progress(base,cursor,metrics)
            del start
        del source
        while cursor<128 and not STOP:
            u.atomic(base/'update-intent.json',dict(next_cursor=cursor+1,epoch_time=time.time()))
            metrics=update(model,ema,bo,ho,reader,lookup,plan['batches'][cursor],c,device);cursor+=1
            write_progress(base,cursor,metrics)
            if cursor%c['save_every']==0 or cursor==128 or STOP:save(base,model,ema,bo,ho,device,cursor,binding,source_ranks)
        if STOP:
            if u.read(base/'latest.json')['cursor']!=cursor:save(base,model,ema,bo,ho,device,cursor,binding,source_ranks)
            u.atomic(base/'interrupted.json',dict(cursor=cursor,block_steps=adam_count(bo),head_steps=adam_count(ho)));return
        # Preserve full update state BEFORE external evaluation can fail.
        before=u.rng_state(device);ema_buffers=u.buffers(ema);(base/'predictions').mkdir(exist_ok=True)
        try:metrics=dict(diagnostic512=u.evaluate(ema,reader,diagnostic,device,base/'predictions/diagnostic512.jsonl.gz'),
                         selection2048=u.evaluate(ema,held_reader,held,device,base/'predictions/selection2048.jsonl.gz'))
        finally:u.restore_rng(before,device);u.restore_buffers(ema,ema_buffers)
        u.atomic(base/'metrics.json',dict(metrics=metrics,fit_updates=128))
        u.atomic(base/'complete.json',dict(status='L27_READOUT128_COMPLETE',cursor=cursor,
             block_steps=adam_count(bo),head_steps=adam_count(ho),checkpoint=u.read(base/'latest.json'),
             metrics=metrics,actor_updates=0,whole_model_lossless_resume=False,epoch_time=time.time(),**binding))
    finally:reader.close();held_reader.close()


def write_progress(base,cursor,metrics):
    row=dict(cursor=cursor,budget=128,epoch_time=time.time(),**metrics)
    u.atomic(base/'progress.json',row)
    with (base/'curve.jsonl').open('a') as f:f.write(u.json.dumps(row)+'\n');f.flush()
    print(u.json.dumps(row),flush=True)


def request_stop(_sig,_frame):
    global STOP
    STOP=True


if __name__=='__main__':main()
