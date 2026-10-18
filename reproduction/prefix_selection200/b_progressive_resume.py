"""Explicit human CP50 rollback, preserving source bytes and every actor clock.

Only B is stopped. The genuine CP50 full-SHA/CPU proof is consumed once; this
adapter restores CP50 rather than pretending discarded51+ optimizer writes
belong to its continuation. It does not change actor, reward or PPO mathematics.
"""
import hashlib,json,os,time,uuid
from pathlib import Path
from reproduction.prefix20_online.launch import read,sha,once,publish
from .b_progressive import MODE,enabled,validate
PROTECTED=('actor.py','actor_safety.py','interaction_protocol.py','grounding.py','ab_probe.py','ab_policy.py','probe_plan.py')
LATEST='same_run_progressive_latest_joint_v1'

def boundary(config):
    """The initial human rollback and later same-run recovery are distinct."""
    spec=config['resume_joint'];n=spec['completed_outer'];latest=spec.get('recovery_mode')==LATEST
    if type(n)is not int or (not latest and n!=50)or(latest and not 51<=n<=200):
        raise ValueError('Initial fork requires50; latest-state recovery requires the actual51..200 boundary')
    return n,latest

def qualified_boundary(config,source,proof,joint):
    """Use actual recorded optimizer clocks, never theoretical20N/10N."""
    n,latest=boundary(config)
    if (proof['status']!='AB_JOINT_FULL_SHA_CPU_PASS'or proof['outer']!=n or not proof['actual_cpu_deserialization']
            or proof['run_id']!=config['run_id']or joint['run_id']!=config['run_id']or joint['outer']!=n
            or joint['actor']['actual_adam_steps_per_rank']!=proof['actor_actual_adam_steps_per_rank']
            or joint['probe']['actual_adam_steps_per_rank']!=proof['probe_actual_adam_steps_per_rank']):
        raise ValueError('Actual latest full-state CPU qualification/actor/probe clocks differ')
    if not latest and joint['actor']['actual_adam_steps_per_rank']!=[1000]*4:
        raise ValueError('Original explicitly authorized CP50 actor clocks differ')
    if latest and read(Path(source)/'latest-joint.json')!=joint:
        raise ValueError('Requested recovery point is older than the actual latest complete joint')
    return n,latest

def pending_pool(source,config,joint,n):
    """Preserve a reserved next question set, but never import unsaved updates."""
    from .pool import reserve_for_outer,state_sha
    source=Path(source);before=read(joint['pool_snapshot']);current=read(source/'question-pool/state.json')
    if current==before:return dict(state=before,files=[])
    if n==200:raise ValueError('Final pool may not reserve outer201')
    root=source/'outers'/('outer-%06d'%(n+1));reservation=read(root/'pool-reservation.json')
    indices,cycles,after=reserve_for_outer(before,config,n+1)
    if (reservation['before_sha256']!=state_sha(before)or reservation['after_sha256']!=state_sha(after)
            or reservation['indices']!=indices or reservation['cycles']!=cycles
            or reservation['after']!=after or current!=after):
        raise ValueError('Latest pending question reservation/cursor/permutation differs')
    # Questions/selection carry authentic IDs and golds, not old model outputs.
    files=[root/'pool-reservation.json']
    for name in ('selection.json','questions.json'):
        p=root/name
        if p.exists():files.append(p)
    return dict(state=after,files=list(map(str,files)))

def verify(config):
    validate(config);p=read(config['source_compatibility'])
    if sha(config['source_compatibility'])!=config['source_compatibility_sha256'] or p['mode']!=MODE:raise ValueError('Explicit source delta proof changed')
    root=Path(config['base'])/'repo'
    for name in PROTECTED:
        rel='reproduction/prefix_selection200/'+name
        if sha(root/rel)!=p['parent_publication']['files_sha256'][rel] or sha(root/rel)!=p['target_publication']['files_sha256'][rel]:raise ValueError('Unrequested actor/reward/probe objective change: '+rel)
    if config['source_commit']!=p['runtime_source_commit']:raise ValueError('Runtime publication differs')
    return p

def stopped(source):
    from reproduction.prefix_only_correctness_ab.launch import birth_states
    from reproduction.a800.owned_processes import same
    cleanup=read(source/'logs/cleanup-status.json');ledger=read(source/'logs/process-ledger.json')
    births=[v for j in ledger['jobs'].values()for v in j.get('ledger',{}).values()]
    if not cleanup.get('cleaned') or birth_states(births) or same(read(source/'preparation/submit-receipt.json')['leader']):raise ValueError('Original B writers still present')
    return cleanup

def plan(base,config=None):
    base=Path(base);c=config or read(base/'preparation/launch-config.json');verify(c);s=c['resume_joint'];source=Path(s['source_base'])
    n,latest=boundary(c)
    if c['run_id']!=read(source/'preparation/launch-config.json')['run_id']:raise ValueError('Recovery must retain this logical run')
    stopped(source);q=read(s['qualification_proof']);jp=source/'outers'/('outer-%06d'%n)/'complete.json';j=read(jp)
    if sha(s['qualification_proof'])!=s['qualification_proof_sha256']or sha(jp)!=s['source_joint_sha256']:raise ValueError('Pinned complete joint/qualification changed')
    qualified_boundary(c,source,q,j)
    # Proof worker read full model/optimizer bytes. Retain those exact immutable
    # file identities here; a stat is not claimed as a second SHA verification.
    for row in q['files']:
        p=Path(row['path']);st=p.stat()
        if (st.st_size,st.st_ino,st.st_mtime_ns)!=(row['bytes'],row['inode'],row['mtime_ns']):raise ValueError('Qualified payload changed: '+str(p))
    pool=pending_pool(source,read(source/'preparation/launch-config.json'),j,n)if latest else None
    return dict(status='B_LATEST_FULL_JOINT_QUALIFIED_READY'if latest else'B_CP50_QUALIFIED_EXPLICIT_ROLLBACK_READY',source_base=str(source),completed_outer=n,next_outer=n+1,source_joint_sha256=sha(jp),phase_reuse={},payload_hash_repeated=False,
        pending_pool=pool,actor_adam=j['actor']['actual_adam_steps_per_rank'],probe_adam=j['probe']['actual_adam_steps_per_rank'],recovery_mode=LATEST if latest else'EXPLICIT_HUMAN_ROLLBACK50')

def validate_imported(base,config):
    return _validate_imported(base,config,plan)

def _validate_imported(base,config,planner):
    result=planner(base,config);b=Path(base);receipt=read(b/'preparation/resume-joint-receipt.json')
    if receipt['source_joint_sha256']!=result['source_joint_sha256']or read(b/'latest-joint.json')['outer']!=result['completed_outer']:raise ValueError('Imported actual complete joint differs')
    j=read(b/'latest-joint.json')
    if read(b/'actor/latest.json')!=j['actor']or read(b/'probe/latest.json')!=j['probe']:raise ValueError('Actor/probe latest receipts not restored')
    expected_pool=result['pending_pool']['state']if result['pending_pool']else read(j['pool_snapshot'])
    if read(b/'question-pool/state.json')!=expected_pool:raise ValueError('Imported pool reservation state differs')
    return result['completed_outer']

def import_complete(base):
    return _import_complete(base,plan,validate_imported,verify,arm='B')

def _import_complete(base,planner,validator,verifier,*,arm):
    b=Path(base);c=read(b/'preparation/launch-config.json');result=planner(b,c)
    if (b/'preparation/resume-joint-receipt.json').exists():validator(b,c);return read(b/'preparation/resume-joint-receipt.json')
    if(b/'preparation/resume-joint-intent.json').exists():raise ValueError('Existing interrupted import must be reconciled, never replayed')
    once(b/'preparation/resume-joint-intent.json',result);source=Path(result['source_base']);n=result['completed_outer'];j=read(source/'outers'/('outer-%06d'%n)/'complete.json')
    def copy(src,dst):
        dst.parent.mkdir(parents=True,exist_ok=True)
        with dst.open('xb')as f:f.write(src.read_bytes())
    for i in range(1,n+1):
        s=source/'outers'/('outer-%06d'%i);target=b/'outers'/s.name
        for name in ('complete.json','pool-state-at-completion.json','grounding-state-at-completion.json','grounding-refresh.json'):
            if (s/name).exists():copy(s/name,target/name)
    for rel in ('question-pool/binding.json','question-pool/provenance.json'):
        if (source/rel).exists():copy(source/rel,b/rel)
    if result['pending_pool']:
        once(b/'question-pool/state.json',result['pending_pool']['state'])
        for src in result['pending_pool']['files']:copy(Path(src),b/Path(src).relative_to(source))
    else:copy(Path(j['pool_snapshot']),b/'question-pool/state.json')
    copy(source/'outers'/('outer-%06d'%n)/'complete.json',b/'latest-joint.json')
    (b/'actor').mkdir();once(b/'actor/latest.json',j['actor']);(b/'probe').mkdir()
    os.link(j['probe_snapshot'],b/'probe/latest.pt');once(b/'probe/latest.json',j['probe'])
    # Preserve optimizer/RNG/sampler from50. No old51+ pending reservation/partial
    # rollout or actual writes are imported into this explicit rollback branch.
    for name in ('initialization-complete.json','initialization-intent.json'):
        if (source/'probe'/name).exists():copy(source/'probe'/name,b/'probe'/name)
    result.update(status=arm+'_LATEST_FULL_JOINT_IMPORTED'if result['pending_pool']is not None else arm+'_CP50_FULL_JOINT_IMPORTED',discarded_later_history_preserved_at_source=True,source_binding=verifier(c),imported_actor_adam=j['actor']['actual_adam_steps_per_rank'],imported_probe_adam=j['probe']['actual_adam_steps_per_rank'],unsaved_optimizer_or_rollout_payloads_imported=False)
    once(b/'preparation/resume-joint-receipt.json',result);validator(b,c);return result


def prepare(source,base,staged_repo,commit,authorization,*,_completed_outer=50,_latest=False):
    """Remote only: code publication plus CP50 state references, no Mac payloads."""
    from .i1_i6_joint_resume import source_manifest
    source,base,repo=map(Path,(source,base,staged_repo));old=read(source/'preparation/launch-config.json');stopped(source)
    n=_completed_outer
    if type(n)is not int or(not _latest and n!=50)or(_latest and not 51<=n<=200):raise ValueError('Explicit initial50 or same-run latest51..200 required')
    if _latest and(read(source/'latest-joint.json')['outer']!=n or not enabled(old)):
        raise ValueError('Latest recovery must follow the actual current progressive B run')
    proof=source/'preparation'/('ab-joint-%04d-independent-full-sha-cpu.json'%n);q=read(proof)
    if q['status']!='AB_JOINT_FULL_SHA_CPU_PASS'or q['outer']!=n:raise ValueError('Actual requested full-state qualification incomplete')
    if base.exists():raise ValueError('Existing new namespace must be reconciled')
    base.mkdir();(base/'preparation').mkdir();(base/'logs').mkdir();(base/'repo').mkdir()
    files=[]
    for p in sorted(repo.rglob('*')):
        if not p.is_file()or'__pycache__'in p.parts:continue
        target=base/'repo'/p.relative_to(repo);target.parent.mkdir(parents=True,exist_ok=True);os.link(p,target)
        if sha(p)!=sha(target):raise ValueError('Target code publication differs')
        files.append(dict(path=p.relative_to(repo).as_posix(),sha256=sha(target)))
    once(base/'preparation/source-full-sha.json',dict(commit=commit,files=files))
    def rebase(value):
        if isinstance(value,dict):return{k:rebase(v)for k,v in value.items()}
        if isinstance(value,list):return[rebase(v)for v in value]
        if isinstance(value,str)and value.startswith(str(source)):return str(base)+value[len(str(source)):]
        return value
    from .b_progressive import prefix,ENV
    length=prefix(n)if _latest else 20
    c=rebase(old);c.update(base=str(base),execution_id=uuid.uuid4().hex,source_commit=commit,science_source_commit=read(source/'outers'/('outer-%06d'%n)/'complete.json')['actor']['binding']['source_commit'],prefix_policy_tokens=length,hidden_capture_contract='sample%d_keep%d_postblock27_native_residual_sum_v1'%(length+1,length),b_progressive=old['b_progressive']if _latest else dict(mode=MODE,resume_outer=50,authorization=authorization))
    if not _latest:c['actor_global_micro_batch_size']=4
    # Remove already-consumed first-outer imports; this execution resumes a joint.
    for key in ('startup_sampling_resume','resume_import','resume_pending_probe','node_migration','source_compatibility','source_compatibility_sha256'):c.pop(key,None)
    c['resume_joint']=dict(adapter='i1_i6_joint_v1',source_base=str(source),completed_outer=n,source_commit=old['source_commit'],source_execution_id=old['execution_id'],qualification_proof=str(proof),qualification_proof_sha256=sha(proof),source_joint_sha256=sha(source/'outers'/('outer-%06d'%n)/'complete.json'))
    if _latest:c['resume_joint']['recovery_mode']=LATEST
    parent=source_manifest(source/'repo',old['source_commit']);target=source_manifest(base/'repo',commit,science_source_commit=c['science_source_commit'])
    compat=dict(mode=MODE,science_source_commit=c['science_source_commit'],runtime_source_commit=commit,parent_publication=parent,target_publication=target,controller_changes={k:dict(parent=parent['files_sha256'].get(k),target=target['files_sha256'].get(k))for k in set(parent['files_sha256'])|set(target['files_sha256'])if parent['files_sha256'].get(k)!=target['files_sha256'].get(k)},authorized_candidate_delta=None,authorized_infrastructure_delta=None)
    path=base/'preparation/b-progressive-source-compatibility.json';once(path,compat);c.update(source_compatibility=str(path),source_compatibility_sha256=sha(path));once(base/'preparation/launch-config.json',c)
    verify(c);result=import_complete(base)
    return dict(status='B_PROGRESSIVE_LATEST_JOINT_PREPARED'if _latest else'B_PROGRESSIVE_CP50_PREPARED',base=str(base),run_id=c['run_id'],commit=commit,files=len(files),config_sha256=sha(base/'preparation/launch-config.json'),imported_boundary=result['completed_outer'],required_process_environment={ENV:str(length)},next_outer=n+1,no_new_actor_probe_or_rng_initialization=True)


def recovery_environment(source):
    """Read JSON first, so the controller can start the right isolated process."""
    from .b_progressive import prefix,ENV
    n=read(Path(source)/'latest-joint.json')['outer']
    if type(n)is not int or not 51<=n<=200:raise ValueError('Actual progressive latest51..200 required')
    return {ENV:str(prefix(n))}


def prepare_latest(source,base,staged_repo,commit):
    """Fault/handoff only: actual stopped source, latest proof, no older fallback.

    This is not a submit/start action. The controller must launch with the
    returned process-scoped prefix environment before Python imports contract.
    """
    import sys
    from .b_progressive import ENV
    source=Path(source);stopped(source);n=read(source/'latest-joint.json')['outer'];expected=recovery_environment(source)
    loaded=sys.modules.get('reproduction.prefix_selection200.contract')
    if os.environ.get(ENV)!=expected[ENV]or(loaded is not None and loaded.PREFIX_TOKENS!=int(expected[ENV])):
        raise RuntimeError('Start a fresh recovery Python process with '+ENV+'='+expected[ENV]+' before importing contract/model modules')
    return prepare(source,base,staged_repo,commit,None,_completed_outer=n,_latest=True)
