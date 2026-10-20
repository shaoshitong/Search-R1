"""Independent durable old+new-head full checkpoint transaction.

No launch, model creation, asset download, or implicit optimizer reset. A joint
latest receipt becomes visible only after immutable file SHA and full CPU reload
of all nine accepted components, head native buffers, and source archive pass.
"""
import copy
import gc
import hashlib
import os
from pathlib import Path
import time
import torch
from . import probe_ablation_train as u
from . import direct_hidden_count as h

ROLE='DIRECT_HIDDEN_COUNT_JOINT_FULL_STATE_V1'
PHASE='accepted_update_complete'
BASE_STEPS=10464


def state_hashes(legacy,legacy_ema,legacy_adam,head,head_ema,head_adam,device):
    old=u.state_hashes(legacy,legacy_ema,legacy_adam,device);new=h.new_head_hashes(head,head_ema,head_adam)
    return dict(legacy_raw=old['raw'],legacy_ema=old['ema'],legacy_adam=old['adam'],
        legacy_raw_buffers=old['raw_buffers'],legacy_ema_buffers=old['ema_buffers'],rng=old['rng'],
        head_raw=new['raw'],head_ema=new['ema'],head_adam=new['adam'])


def _tree_hash(value):
    """Storage-identity-independent full original rank archive fingerprint."""
    sha=hashlib.sha256()
    def visit(v):
        if torch.is_tensor(v):
            sha.update(('tensor:'+str(v.dtype)+':'+str(tuple(v.shape))).encode())
            sha.update(v.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        elif isinstance(v,u.np.ndarray):
            sha.update(('numpy:'+str(v.dtype)+':'+str(v.shape)).encode());sha.update(v.tobytes())
        elif isinstance(v,dict):
            sha.update(b'dict')
            for key in sorted(v,key=lambda k:str(k)):visit(key);visit(v[key])
        elif isinstance(v,(tuple,list)):
            sha.update(type(v).__name__.encode())
            for item in v:visit(item)
        elif v is None or isinstance(v,(str,bytes,bool,int,float,u.np.generic)):
            sha.update((type(v).__name__+':'+repr(v)).encode())
        else:raise ValueError('Unsupported historical archive state type: '+str(type(v)))
    visit(value);return sha.hexdigest()


def _archive(archive):
    if (not isinstance(archive,list) or len(archive)!=4 or
            sorted(s['rank'] for s in archive)!=[0,1,2,3] or any('rng' not in s for s in archive)):
        raise ValueError('All four original source rank RNG records required')
    return _tree_hash(archive)


def _clock(optimizer):
    values={int(v['step']) for v in optimizer['state'].values() if v}
    if not values:return 0
    if len(values)!=1:raise ValueError('Saved Adam counters disagree')
    return values.pop()


def _binding(binding):
    if not isinstance(binding,dict) or not binding:raise ValueError('Complete explicit config binding required')
    return u.digest(binding)


def _finite(mapping):
    for value in mapping.values():
        if not torch.is_tensor(value) or not bool(torch.isfinite(value).all()):raise ValueError('Nonfinite saved parameters or buffers')


def _saved_hashes(blob):
    old=blob['legacy'];new=blob['head']
    return dict(legacy_raw=u.model_mapping_hash(old['raw']),legacy_ema=u.model_mapping_hash(old['ema']),
        legacy_adam=u.optimizer_blob_hash(old['optimizer']),legacy_raw_buffers=u.tensor_hash(old['raw_buffers']),
        legacy_ema_buffers=u.tensor_hash(old['ema_buffers']),rng=u.rng_hash(old['rng']),
        head_raw=u.model_mapping_hash(new['raw']),head_ema=u.model_mapping_hash(new['ema']),
        head_adam=u.optimizer_blob_hash(new['optimizer']))


def _validate(blob,expected_binding=None,expected_cursor=None):
    if blob.get('role')!=ROLE or blob.get('phase')!=PHASE:raise ValueError('Unknown joint checkpoint role/phase')
    cursor=blob['cursor']
    if type(cursor) is not int or cursor<0 or (expected_cursor is not None and cursor!=expected_cursor):raise ValueError('Joint data cursor differs')
    binding=blob['binding'];bound=_binding(binding)
    if bound!=blob['binding_sha256'] or (expected_binding is not None and _binding(expected_binding)!=bound):raise ValueError('Joint immutable config binding differs')
    data=blob['data_cursor']
    if (not isinstance(data,dict) or data.get('cursor')!=cursor or not isinstance(data.get('plan_sha256'),str)
            or len(data['plan_sha256'])!=64 or any(c not in '0123456789abcdef' for c in data['plan_sha256'])):
        raise ValueError('Explicit accepted cursor and immutable plan SHA required')
    if isinstance(binding.get('plan'),dict) and binding['plan'].get('sha256')!=data['plan_sha256']:raise ValueError('Data cursor plan differs from config')
    old,new=blob['legacy'],blob['head']
    if (old['cursor']!=cursor or old['steps']!=BASE_STEPS+cursor or _clock(old['optimizer'])!=BASE_STEPS+cursor
            or new['steps']!=cursor or _clock(new['optimizer'])!=cursor):raise ValueError('Inherited/new Adam clocks must remain 10464+cursor and cursor')
    if (old['source_world_size']!=4 or old['current_world_size']!=1 or old['hardware_topology_changed'] is not True
            or old['four_rank_bitexact_resume_claim'] is not False or old['actor_updates']!=0):
        raise ValueError('Truthful E1 four-rank archive to single-GPU branch provenance required')
    if old['config_sha256']!=bound or old['matched_binding_sha256']!=bound:raise ValueError('Legacy nested binding differs')
    if _archive(old['source_all_rank_states'])!=blob['source_archive_sha256']:raise ValueError('Original source archive changed')
    for saved in (old,new):
        for key in ('raw','ema'):_finite(saved[key])
        for values in saved['optimizer']['state'].values():
            for key in ('exp_avg','exp_avg_sq'):
                if key not in values or not bool(torch.isfinite(values[key]).all()):raise ValueError('Invalid saved old/new Adam moments')
    for key in ('raw_buffers','ema_buffers'):
        _finite(old[key]);_finite(blob['head_'+key])
    hashes=_saved_hashes(blob)
    if hashes!=blob['accepted_hashes']:raise ValueError('Joint nine-component checksum differs')
    if new['accepted_hashes']!=dict(raw=hashes['head_raw'],ema=hashes['head_ema'],adam=hashes['head_adam']):raise ValueError('Nested new-head hashes differ')
    old_expected=dict(raw=hashes['legacy_raw'],ema=hashes['legacy_ema'],adam=hashes['legacy_adam'],
        raw_buffers=hashes['legacy_raw_buffers'],ema_buffers=hashes['legacy_ema_buffers'],rng=hashes['rng'])
    if old_expected!=old['accepted_hashes']:raise ValueError('Nested legacy six-component checksum differs')
    actual_buffers=dict(raw=u.tensor_hash(blob['head_raw_buffers']),ema=u.tensor_hash(blob['head_ema_buffers']))
    if actual_buffers!=blob['head_buffer_hashes']:raise ValueError('New-head native buffers changed')
    return hashes


def joint_blob(legacy,legacy_ema,legacy_adam,head,head_ema,head_adam,device,cursor,binding,
        source_rank_states,data_cursor,phase=PHASE):
    bound=_binding(binding)
    old=u.checkpoint_blob(legacy,legacy_ema,legacy_adam,device,cursor,
        dict(config_sha256=bound,matched_binding_sha256=bound),copy.deepcopy(source_rank_states))
    blob=dict(role=ROLE,phase=phase,cursor=cursor,binding=copy.deepcopy(binding),binding_sha256=bound,
        data_cursor=copy.deepcopy(data_cursor),legacy=old,head=h.new_head_state(head,head_ema,head_adam),
        head_raw_buffers=u.buffers(head),head_ema_buffers=u.buffers(head_ema),
        head_buffer_hashes=dict(raw=u.tensor_hash(u.buffers(head)),ema=u.tensor_hash(u.buffers(head_ema))),
        accepted_hashes=state_hashes(legacy,legacy_ema,legacy_adam,head,head_ema,head_adam,device),
        source_archive_sha256=_archive(source_rank_states),old_optimizer_reset=False,
        new_head_optimizer_initialized_once=True,actor_updates=0,evaluation_started=False)
    _validate(blob);return blob


def restore_joint(blob,legacy,legacy_ema,legacy_adam,head,head_ema,head_adam,device,*,
        expected_binding=None,expected_cursor=None):
    expected=_validate(blob,expected_binding,expected_cursor)
    u.restore_blob(blob['legacy'],legacy,legacy_ema,legacy_adam,device)
    h.restore_new_head(blob['head'],head,head_ema,head_adam,device)
    u.restore_buffers(head,blob['head_raw_buffers']);u.restore_buffers(head_ema,blob['head_ema_buffers'])
    u.restore_rng(blob['legacy']['rng'],device)
    actual=state_hashes(legacy,legacy_ema,legacy_adam,head,head_ema,head_adam,device)
    if actual!=expected:raise ValueError('Joint in-memory full restore differs')
    if (u.tensor_hash(u.buffers(head))!=blob['head_buffer_hashes']['raw'] or
            u.tensor_hash(u.buffers(head_ema))!=blob['head_buffer_hashes']['ema']):raise ValueError('Restored new-head buffers differ')
    return dict(cursor=blob['cursor'],data_cursor=copy.deepcopy(blob['data_cursor']),phase=blob['phase'],
        legacy_Adam_steps=u.steps(legacy_adam),new_head_Adam_steps=h.optimizer_steps(head_adam),
        accepted_hashes=actual,source_archive_sha256=blob['source_archive_sha256'])


def verify_joint_file(path,*,expected_binding=None,expected_cursor=None,expected_sha256=None):
    """CPU-only standalone full-file and nine-component deserialization gate."""
    path=Path(path);st=path.stat();sha=u.sha(path)
    if expected_sha256 is not None and sha!=expected_sha256:raise ValueError('Joint full-file SHA differs')
    saved=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
    hashes=_validate(saved,expected_binding,expected_cursor)
    result=dict(path=str(path),bytes=st.st_size,sha256=sha,cursor=saved['cursor'],
        steps=saved['legacy']['steps'],legacy_Adam_steps=saved['legacy']['steps'],
        new_head_Adam_steps=saved['head']['steps'],config_sha256=saved['binding_sha256'],
        matched_binding_sha256=saved['binding_sha256'],data_cursor=copy.deepcopy(saved['data_cursor']),
        phase=saved['phase'],accepted_hashes=hashes,head_buffer_hashes=saved['head_buffer_hashes'],
        source_archive_sha256=saved['source_archive_sha256'],source_all_four_RNG_preserved=True,
        full_CPU_reload_passed=True,full_file_SHA256_verified=True,actor_updates=0,epoch_time=time.time())
    del saved;gc.collect()
    if (st.st_size,st.st_mtime_ns)!=(path.stat().st_size,path.stat().st_mtime_ns):raise ValueError('Joint checkpoint changed during full CPU verification')
    return result


def _fsync_dir(path):
    fd=os.open(str(path),os.O_RDONLY|getattr(os,'O_DIRECTORY',0))
    try:os.fsync(fd)
    finally:os.close(fd)


def save_joint(base,blob):
    """Durable immutable file, fresh SHA, complete CPU reload, THEN latest."""
    base=Path(base);root=base/'checkpoints';root.mkdir(parents=True,exist_ok=True);_validate(blob)
    destination=root/('cursor-%06d.pt'%blob['cursor']);temporary=destination.with_suffix('.unaccepted.tmp')
    if destination.exists() or temporary.exists():raise ValueError('Preserve existing accepted or partial joint checkpoint')
    with temporary.open('xb') as stream:torch.save(blob,stream);stream.flush();os.fsync(stream.fileno())
    temporary.replace(destination);_fsync_dir(root)
    receipt=verify_joint_file(destination,expected_binding=blob['binding'],expected_cursor=blob['cursor'])
    u.atomic(base/'latest.json',receipt);_fsync_dir(base)
    return receipt
