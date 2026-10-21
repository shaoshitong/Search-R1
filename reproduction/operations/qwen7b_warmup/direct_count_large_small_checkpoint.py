"""Independent CPU-only small new-head CP1 acceptance; no legacy model load.

Run only in the owned allocation with CUDA_VISIBLE_DEVICES empty. Size/ZIP/meta
geometry guards precede CPU payload loading. Archived E1 RNG/digest provenance
is checked against this allocation's small accepted CP0, not old model weights.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import zipfile

LIMIT=64*1024**2
E1_SHA='02884a3c76e88d1014c4876e1ee2ff4b4405e7f94de7b9112881fed183862724'
ROOT_KEYS={'kind','head','rng','cursor','epoch','config','cache','source_all_four_rank_states','frozen_legacy_hashes','initial_head_hashes','accepted_hashes','data_cursor'}


def proof(path):
    p=Path(path);h=hashlib.sha256()
    with p.open('rb')as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())


def checked(ref):
    if proof(ref['path'])!={k:ref[k]for k in ('path','bytes','sha256')}:raise ValueError('Small immutable source changed')
    return Path(ref['path'])


def read(path):return json.loads(Path(path).read_bytes())


def immutable(path,value):
    p=Path(path);raw=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()+b'\n';p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():
        if p.read_bytes()!=raw:raise ValueError('Existing independent CP1 receipt differs')
    else:
        tmp=p.with_name(p.name+'.tmp');tmp.write_bytes(raw);tmp.replace(p)


def zip_guard(path,limit=LIMIT):
    p=Path(path);size=p.stat().st_size
    if not 0<size<=limit:raise ValueError('Only small new-head checkpoints are allowed; legacy weights forbidden')
    with zipfile.ZipFile(p)as z:
        entries=z.infolist()
        if len(entries)>128 or sum(x.file_size for x in entries)>limit or any(x.compress_type!=zipfile.ZIP_STORED or '..'in Path(x.filename).parts for x in entries):raise ValueError('Unexpected checkpoint storage range')
        pickles=[x for x in entries if x.filename.endswith('/data.pkl')]
        if len(pickles)!=1 or pickles[0].file_size>512*1024:raise ValueError('Bounded checkpoint metadata required')
        stores=[x for x in entries if '/data/'in x.filename]
        if not stores or any(x.file_size>2*1024**2 for x in stores):raise ValueError('Unexpected large tensor storage; no legacy model load')
    return dict(file_bytes=size,storage_entries=len(stores),uncompressed_tensor_bytes=sum(x.file_size for x in stores),largest_storage_bytes=max(x.file_size for x in stores))


def shapes(dimension=3584,embedding=128):
    return {'attention.weight':(1,dimension),'projection.weight':(embedding,dimension),'projection.bias':(embedding,),
        'classifier.0.weight':(128,3*embedding),'classifier.0.bias':(128,),'classifier.2.weight':(6,128),'classifier.2.bias':(6,)}


def geometry(state,expected=None):
    import torch
    expected=expected or shapes()
    if set(state)!=ROOT_KEYS or state['kind']!='LARGE_FROZEN_E1_HEAD_ONLY_FULL_STATE':raise ValueError('Only isolated new-head schema accepted')
    h=state['head']
    if set(h)!={'raw','ema','optimizer','steps','accepted_hashes','new_parameters_initialized_once','inherited_legacy_optimizer_reset'}:raise ValueError('Unexpected head state scope')
    if h['new_parameters_initialized_once']is not True or h['inherited_legacy_optimizer_reset']is not False:raise ValueError('Fresh new-head / inherited frozen legacy provenance differs')
    for name in ('raw','ema'):
        if set(h[name])!=set(expected):raise ValueError('Only seven new-head parameter tensors accepted')
        for key,value in h[name].items():
            if not torch.is_tensor(value)or tuple(value.shape)!=expected[key]or value.dtype!=torch.float32:raise ValueError('Unexpected new-head tensor shape/dtype')
    opt=h['optimizer'];groups=opt['param_groups']
    if set(opt)!={'state','param_groups'}or len(groups)!=1 or len(groups[0]['params'])!=7 or len(set(groups[0]['params']))!=7:raise ValueError('Exactly seven new-head Adam parameter slots required')
    if h['steps']not in (0,1):raise ValueError('Only initial CP0 and first update CP1 accepted')
    if h['steps']==0 and opt['state']:raise ValueError('Initial new Adam must be empty')
    if h['steps']==1:
        if set(opt['state'])!=set(groups[0]['params']):raise ValueError('Every new-head Adam slot must be populated')
        for parameter,key in zip(groups[0]['params'],h['raw']):
            item=opt['state'][parameter]
            if set(item)!={'step','exp_avg','exp_avg_sq'}:raise ValueError('Unexpected new-head Adam contents')
            for field in ('exp_avg','exp_avg_sq'):
                if tuple(item[field].shape)!=expected[key]or item[field].dtype!=torch.float32:raise ValueError('Adam tensor geometry differs')
            if not torch.is_tensor(item['step'])or item['step'].numel()!=1:raise ValueError('Scalar actual Adam counter required')
    archive=state['source_all_four_rank_states']
    if len(archive)!=4 or sorted(x['rank']for x in archive)!=[0,1,2,3]:raise ValueError('Four original E1 rank archives required')
    for item in archive:
        if set(item)!={'rank','model_sha256','ema_sha256','optimizer_sha256','steps','rng'}:raise ValueError('Only source rank digests/RNG may be archived; model/Adam payload forbidden')
        if item['steps']!=10464 or item['model_sha256']!=state['frozen_legacy_hashes']['raw']:raise ValueError('Original E1 archive identity/counter differs')
    return dict(new_head_parameters=sum(v.numel()for v in h['raw'].values()),new_head_parameter_tensors=7,old_model_parameter_tensors=0,old_optimizer_tensors=0)


def mapping_hash(mapping):
    h=hashlib.sha256()
    for key,value in mapping.items():h.update(key.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def adam_hash(opt):
    import torch
    h=hashlib.sha256()
    for group in opt['param_groups']:
        for parameter in group['params']:
            for key,value in sorted(opt['state'].get(parameter,{}).items()):
                h.update(key.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes()if torch.is_tensor(value)else repr(value).encode())
    return h.hexdigest()


def rng_hash(value):
    import numpy as np
    import torch
    if set(value)!={'python','numpy','torch','cuda'}:raise ValueError('Complete Python/NumPy/torch/CUDA stored RNG required')
    py=random.Random();py.setstate(value['python']);n=value['numpy'];numpy_rng=np.random.RandomState();numpy_rng.set_state(n)
    back=numpy_rng.get_state()
    if py.getstate()!=value['python']or back[0]!=n[0]or not np.array_equal(back[1],n[1])or back[2:]!=n[2:]:raise ValueError('Python/NumPy RNG roundtrip differs')
    for key in ('torch','cuda'):
        v=value[key]
        if v is not None and (not torch.is_tensor(v)or v.dtype!=torch.uint8 or v.device.type!='cpu'or v.ndim!=1):raise ValueError('Stored RNG must be CPU bytes')
    if value['torch']is None:raise ValueError('CPU torch RNG missing')
    generator=torch.Generator(device='cpu');generator.set_state(value['torch'])
    if not torch.equal(generator.get_state(),value['torch']):raise ValueError('CPU RNG roundtrip failed')
    h=hashlib.sha256(repr(value['python']).encode());h.update(n[0].encode());h.update(n[1].tobytes());h.update(repr(n[2:]).encode())
    for name in ('torch','cuda'):h.update(name.encode());h.update(value[name].numpy().tobytes()if value[name]is not None else b'None')
    return h.hexdigest()


def component_hashes(state):
    import torch
    h=state['head']
    for values in (h['raw'],h['ema']):
        if not all(bool(torch.isfinite(v).all())for v in values.values()):raise ValueError('Nonfinite new head')
    for item in h['optimizer']['state'].values():
        if int(item['step'])!=h['steps']or not all(bool(torch.isfinite(item[k]).all())for k in ('exp_avg','exp_avg_sq'))or not bool((item['exp_avg_sq']>=0).all()):raise ValueError('Adam counter/finite moment mismatch')
    result=dict(raw=mapping_hash(h['raw']),ema=mapping_hash(h['ema']),adam=adam_hash(h['optimizer']),rng=rng_hash(state['rng']))
    if result!=state['accepted_hashes']or {k:result[k]for k in ('raw','ema','adam')}!=h['accepted_hashes']:raise ValueError('Independent four component hashes differ')
    return result


def restore_small(state,dimension=3584,embedding=128):
    import torch
    from torch import nn
    class Head(nn.Module):
        def __init__(self):
            super().__init__();self.attention=nn.Linear(dimension,1,bias=False);self.projection=nn.Linear(dimension,embedding);self.classifier=nn.Sequential(nn.Linear(3*embedding,128),nn.GELU(),nn.Linear(128,6))
    raw,ema=Head(),Head();raw.load_state_dict(state['head']['raw'],strict=True);ema.load_state_dict(state['head']['ema'],strict=True)
    opt=torch.optim.Adam(raw.parameters(),lr=1e-4);opt.load_state_dict(state['head']['optimizer'])
    if mapping_hash(raw.state_dict())!=state['accepted_hashes']['raw']or mapping_hash(ema.state_dict())!=state['accepted_hashes']['ema']or adam_hash(opt.state_dict())!=state['accepted_hashes']['adam']:raise ValueError('Independent small head/Adam CPU restore differs')
    return True


def archive_signature(state):
    rows=[]
    for item in sorted(state['source_all_four_rank_states'],key=lambda x:x['rank']):
        rows.append({**{k:item[k]for k in ('rank','model_sha256','ema_sha256','optimizer_sha256','steps')},'rng_sha256':rng_hash(item['rng'])})
    if any(len({r[k]for r in rows})!=1 for k in ('model_sha256','ema_sha256','optimizer_sha256')):raise ValueError('Original four rank model/Adam/EMA digests disagree')
    return hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def audit(base,expected_config_sha,output):
    import os
    import torch
    base=Path(base)
    if os.environ.get('SLURM_JOB_ID')!='743978'or os.environ.get('CUDA_VISIBLE_DEVICES')!=''or torch.cuda.is_initialized():raise ValueError('Only owned743978 CPU-only allocation step accepted')
    torch.set_num_threads(2)
    config_ref=proof(base/'config.json');c=read(config_ref['path'])
    if config_ref['sha256']!=expected_config_sha or c['source_checkpoint']['sha256']!=E1_SHA or c['legacy_frozen']is not True or c['legacy_updates']!=0 or c['actor_updates']!=0:raise ValueError('Exact frozen E1 config required')
    files=[base/'checkpoints/cursor-000000.pt',base/'checkpoints/cursor-000001.pt']
    if not all(p.exists()for p in files):return dict(status='SMALL_CP1_NOT_YET_PRESENT',actual_CP1_loaded=False,GPU=0)
    ranges=[zip_guard(p)for p in files];refs=[proof(p)for p in files]
    # Meta first enforces the exact model tensor scope before CPU tensor materialization.
    for p in files:geometry(torch.load(p,map_location='meta',mmap=True,weights_only=False))
    states=[torch.load(p,map_location='cpu',mmap=True,weights_only=False)for p in files]
    checks=[]
    cache_ref=proof(base/'cache/complete.json');cache=read(cache_ref['path'])
    if cache['config_sha256']!=expected_config_sha or cache['legacy_updates']!=0 or cache['actor_updates']!=0 or cache['frozen_legacy_before']!=cache['frozen_legacy_after']:raise ValueError('Frozen cache provenance differs')
    for index,state in enumerate(states):
        spec=geometry(state)
        if spec['new_head_parameters']!=512518 or state['cursor']!=index or state['head']['steps']!=index or state['epoch']!=0 or state['config']!=config_ref or state['cache']!=cache_ref or state['data_cursor']!=dict(cursor=index,orders_sha256=c['orders']['sha256'])or state['frozen_legacy_hashes']!=cache['frozen_legacy_before']:raise ValueError('Small checkpoint config/cache/data/Adam cursor differs')
        group=state['head']['optimizer']['param_groups'][0]
        if group['lr']!=c['head_LR']or group['weight_decay']!=c['head_weight_decay']:raise ValueError('Head Adam hyperparameters differ')
        if state['rng']['cuda']is None or any(item['rng']['cuda']is None for item in state['source_all_four_rank_states']):raise ValueError('Actual GPU training/source CUDA RNG bytes missing')
        value=component_hashes(state);restore_small(state);checks.append(dict(cursor=index,components=value,geometry=spec))
    zero,one=states
    if archive_signature(zero)!=archive_signature(one)or zero['initial_head_hashes']!=one['initial_head_hashes']or zero['initial_head_hashes']!={k:checks[0]['components'][k]for k in ('raw','ema','adam')}:raise ValueError('Original source archive or once-only head initialization changed')
    replay_ref=proof(base/'first-update-replay.json');replay=read(replay_ref['path'])
    if replay['status']!='PASSED'or replay['Adam_before']!=0 or replay['Adam_after']!=1 or replay['expected']!=replay['actual']or replay['actual']!=checks[1]['components']:raise ValueError('Actual first update replay/CP1 components differ')
    if [proof(p)for p in files]!=refs or torch.cuda.is_initialized():raise ValueError('Checkpoint changed or CUDA unexpectedly initialized')
    value=dict(status='INDEPENDENT_SMALL_HEAD_CP1_FOUR_COMPONENTS_CPU_RELOAD_PASSED',config=config_ref,cache=cache_ref,
        implementation=proof(__file__),CP0=refs[0],CP1=refs[1],zip_ranges=ranges,components=checks[1]['components'],new_head_parameters=512518,
        first_update_replay=replay_ref,head_Adam_steps=1,data_cursor=one['data_cursor'],source_E1_descriptor=c['source_checkpoint'],
        archived_four_rank_signature_sha256=archive_signature(one),archived_rank_steps=[10464]*4,archive_unchanged_from_small_CP0=True,
        old_E1_model_or_Adam_payload_opened=False,old_model_tensor_count=0,new_head_raw_EMA_Adam_restored_CPU=True,
        Python_NumPy_torch_CPU_RNG_restored=True,stored_CUDA_RNG_bytes_hashed=True,CUDA_RNG_kernel_restore_claim=False,
        legacy_or_actor_updates=0,audit_optimizer_updates=0,CPU_threads=2,GPU=0,CUDA_initialized=False)
    immutable(output,value)
    return dict(value,receipt=proof(output))


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--config-sha256',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    print(json.dumps(audit(a.base,a.config_sha256,a.output),sort_keys=True),flush=True)


if __name__=='__main__':main()
