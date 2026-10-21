"""Native joint six-state checkpoints with immutable pins and one rolling body.

The inherited saver performs full CPU raw/EMA/Adam/buffer/RNG and file-SHA
acceptance in an isolated staging directory. latest.json points to a stable
accepted file; rolling-latest.pt is a hard-link alias. Thus replacing that alias
never invalidates the previously published latest receipt. Previous rolling
storage is removed only after the new latest receipt is durably published.
No training, restore, optimizer reset, or launch is performed here.
"""
import fcntl,gc,json,os,shutil,time,uuid
from pathlib import Path
import torch
from . import probe_ablation_train as u

ROLE='BALANCED_HARD_REAL_G10_NATIVE_JOINT_E4_CONTINUATION'
SOURCE_STEPS=14418


def _metadata(state):
    cursor=state.get('cursor');sampler=state.get('sampler_state');data=state.get('data_cursor')
    if (state.get('role')!=ROLE or state.get('phase')!='accepted_update_complete'
        or type(cursor)is not int or cursor<0 or state.get('source_steps')!=SOURCE_STEPS
        or state.get('steps')!=SOURCE_STEPS+cursor):
        raise ValueError('Exact native accepted optimizer/cursor phase required')
    if (not isinstance(sampler,dict) or sampler.get('accepted_updates')!=cursor
        or not isinstance(sampler.get('binding'),str) or not sampler['binding']
        or u.digest(sampler)!=state.get('sampler_state_sha256')):
        raise ValueError('Native sampler state checksum/cursor differs')
    if (not isinstance(data,dict) or data.get('cursor')!=cursor
        or data.get('sampler_binding')!=sampler['binding']
        or data.get('sampler_state_sha256')!=state['sampler_state_sha256']):
        raise ValueError('Explicit data cursor/sampler binding differs')
    archive=state.get('source_all_rank_states')
    if not isinstance(archive,list) or len(archive)!=4 or sorted(s.get('rank')for s in archive)!=[0,1,2,3] or any('rng'not in s for s in archive):
        raise ValueError('Original four-rank RNG archive is incomplete')
    for key in ('config_sha256','matched_binding_sha256'):
        if not isinstance(state.get(key),str) or not state[key]:raise ValueError('Full immutable config binding required')
    return dict(cursor=cursor,steps=state['steps'],config_sha256=state['config_sha256'],
        matched_binding_sha256=state['matched_binding_sha256'],sampler_state_sha256=state['sampler_state_sha256'],
        data_cursor=data,accepted_hashes=state['accepted_hashes'])


def _fsync_dir(path):
    fd=os.open(str(path),os.O_RDONLY|getattr(os,'O_DIRECTORY',0))
    try:os.fsync(fd)
    finally:os.close(fd)


def _same(receipt,expected):
    return all(receipt.get(k)==v for k,v in expected.items())


def _checked_previous(base,expected):
    latest=base/'latest.json'
    if not latest.exists():return None
    prior=u.read(latest)
    if prior['cursor']>expected['cursor']:raise ValueError('Refusing checkpoint cursor rollback')
    if any(prior[k]!=expected[k]for k in('config_sha256','matched_binding_sha256')):
        raise ValueError('Existing latest belongs to different native training binding')
    if not Path(prior['path']).is_file():raise ValueError('Previous accepted latest checkpoint is missing')
    return prior


def save_native(base,state,pin=False):
    """Save/accept a full blob in the current owned allocation; return u-style ref.

    An interrupted publish leaves its durable transaction and accepted new body
    for explicit reconciliation. It never deletes the previous latest body or
    silently advances past an unresolved transaction.
    """
    if not os.environ.get('SLURM_JOB_ID'):raise ValueError('Native checkpoint I/O requires the owned Slurm allocation')
    base=Path(base);base.mkdir(parents=True,exist_ok=True);root=base/'checkpoints';root.mkdir(exist_ok=True)
    expected=_metadata(state);guard=(base/'native-checkpoint.lock').open('a')
    with guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        journal=base/'native-checkpoint-transaction.json'
        if journal.exists():
            old=u.read(journal)
            if old.get('phase')!='published':raise ValueError('Unresolved native checkpoint publish transaction; reconcile accepted files first')
        previous=_checked_previous(base,expected)
        # Exact repeat calls reuse the already accepted immutable file. Verify
        # SHA before returning; never overwrite a pin or manufacture acceptance.
        if previous and previous['cursor']==expected['cursor']:
            if not _same(previous,expected):raise ValueError('Same native cursor has different full state or sampler')
            u.checked(previous)
            if not pin or previous.get('pinned') is True:return previous
        stage=Path(os.path.realpath(root))/('.native-stage-'+uuid.uuid4().hex);stage.mkdir()
        published=False;body=None
        try:
            accepted=u.save_checkpoint(stage,state)
            staged=Path(accepted['path'])
            # The inherited saver has already read all six components and SHA.
            # This second mmap touches only native sampler/cursor metadata.
            loaded=torch.load(staged,map_location='cpu',mmap=True,weights_only=False)
            if _metadata(loaded)!=expected:raise ValueError('Saved native sampler/data cursor does not match accepted state')
            del loaded;gc.collect()
            if not _same(accepted,{k:expected[k]for k in('cursor','steps','config_sha256','matched_binding_sha256','accepted_hashes')}):
                raise ValueError('Inherited full checkpoint receipt differs')
            if pin:
                body=root/('cursor-%06d.pt'%expected['cursor'])
                if body.exists():
                    pinref=body.with_suffix('.json')
                    if not pinref.exists() or not _same(u.read(pinref),expected):raise ValueError('Immutable pin already exists with unknown/different state')
                    priorpin=u.read(pinref);u.checked(priorpin);staged.unlink();accepted=dict(priorpin)
                else:os.link(staged,body);staged.unlink()
            else:
                body=root/('.rolling-accepted-cursor-%06d-%s.pt'%(expected['cursor'],uuid.uuid4().hex))
                os.replace(staged,body)
            receipt=dict(accepted,**expected,path=str(body.absolute()),bytes=body.stat().st_size,
                pinned=bool(pin),full_CPU_six_state_reload_passed=True,sampler_full_state_and_cursor_reload_passed=True,
                saved_all_four_source_RNG_states=True,checkpoint_kind='native_joint_pin'if pin else'native_joint_rolling',
                allocation_job_id=os.environ['SLURM_JOB_ID'],epoch_time=time.time())
            # Data were unchanged by link/rename. The accepted full-file SHA is
            # inherited; the stable body remains available through publication.
            _fsync_dir(root)
            if pin:u.atomic(body.with_suffix('.json'),receipt)
            u.atomic(journal,dict(phase='verified_new_body_before_publish',new_checkpoint=receipt,previous_checkpoint=previous,epoch=time.time()))
            alias=root/'rolling-latest.pt';temporary=root/('.rolling-link-'+uuid.uuid4().hex)
            os.link(body,temporary);os.replace(temporary,alias);_fsync_dir(root)
            u.atomic(base/'latest.json',receipt);_fsync_dir(base)
            u.atomic(journal,dict(phase='published',new_checkpoint=receipt,previous_checkpoint=previous,epoch=time.time()));_fsync_dir(base)
            published=True
            if previous:
                old=Path(previous['path'])
                if old!=body and old.parent.absolute()==root.absolute() and old.name.startswith('.rolling-accepted-cursor-'):
                    old.unlink(missing_ok=True);_fsync_dir(root)
            shutil.rmtree(stage)
            return receipt
        except BaseException:
            # Only unpublished owned staging/body files are retained. An old
            # latest receipt still points to its unchanged stable accepted body.
            if not published:
                failure=stage/'failure.json'
                u.atomic(failure,dict(stage='native_checkpoint_not_published',cursor=expected['cursor'],accepted_new_body=str(body)if body else None,old_latest_preserved=True,epoch=time.time()))
            raise
