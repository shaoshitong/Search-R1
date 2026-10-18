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
    if s['completed_outer']!=50 or c['run_id']!=read(source/'preparation/launch-config.json')['run_id']:raise ValueError('Initial B fork must resume actual CP50')
    stopped(source);q=read(s['qualification_proof']);j=read(source/'outers/outer-000050/complete.json')
    if sha(s['qualification_proof'])!=s['qualification_proof_sha256'] or q['status']!='AB_JOINT_FULL_SHA_CPU_PASS' or q['outer']!=50 or not q['actual_cpu_deserialization']:raise ValueError('Actual CP50 qualification missing')
    if j['actor']['actual_adam_steps_per_rank']!=[1000]*4 or j['probe']['actual_adam_steps_per_rank']!=q['probe_actual_adam_steps_per_rank']:raise ValueError('CP50 actor/probe clocks differ')
    # Proof worker read full model/optimizer bytes. Retain those exact immutable
    # file identities here; a stat is not claimed as a second SHA verification.
    for row in q['files']:
        p=Path(row['path']);st=p.stat()
        if (st.st_size,st.st_ino,st.st_mtime_ns)!=(row['bytes'],row['inode'],row['mtime_ns']):raise ValueError('Qualified payload changed: '+str(p))
    return dict(status='B_CP50_QUALIFIED_EXPLICIT_ROLLBACK_READY',source_base=str(source),completed_outer=50,next_outer=51,source_joint_sha256=sha(source/'outers/outer-000050/complete.json'),phase_reuse={},payload_hash_repeated=False)

def validate_imported(base,config):
    result=plan(base,config);b=Path(base);receipt=read(b/'preparation/resume-joint-receipt.json')
    if receipt['source_joint_sha256']!=result['source_joint_sha256'] or read(b/'latest-joint.json')['outer']!=50:raise ValueError('Imported CP50 joint differs')
    if read(b/'actor/latest.json')!=read(b/'latest-joint.json')['actor']:raise ValueError('Actor50 not restored')
    return 50

def import_complete(base):
    b=Path(base);c=read(b/'preparation/launch-config.json');result=plan(b,c)
    if (b/'preparation/resume-joint-receipt.json').exists():validate_imported(b,c);return read(b/'preparation/resume-joint-receipt.json')
    once(b/'preparation/resume-joint-intent.json',result);source=Path(result['source_base']);j=read(source/'outers/outer-000050/complete.json')
    def copy(src,dst):
        dst.parent.mkdir(parents=True,exist_ok=True)
        with dst.open('xb')as f:f.write(src.read_bytes())
    for n in range(1,51):
        s=source/'outers'/('outer-%06d'%n);target=b/'outers'/s.name
        for name in ('complete.json','pool-state-at-completion.json','grounding-state-at-completion.json','grounding-refresh.json'):
            if (s/name).exists():copy(s/name,target/name)
    for rel in ('question-pool/binding.json','question-pool/provenance.json'):
        if (source/rel).exists():copy(source/rel,b/rel)
    copy(Path(j['pool_snapshot']),b/'question-pool/state.json')
    copy(source/'outers/outer-000050/complete.json',b/'latest-joint.json')
    (b/'actor').mkdir();once(b/'actor/latest.json',j['actor']);(b/'probe').mkdir()
    os.link(j['probe_snapshot'],b/'probe/latest.pt');once(b/'probe/latest.json',j['probe'])
    # Preserve optimizer/RNG/sampler from50. No old51+ pending reservation/partial
    # rollout or actual writes are imported into this explicit rollback branch.
    for name in ('initialization-complete.json','initialization-intent.json'):
        if (source/'probe'/name).exists():copy(source/'probe'/name,b/'probe'/name)
    result.update(status='B_CP50_FULL_JOINT_IMPORTED',discarded_later_history_preserved_at_source=True,source_binding=verify(c),imported_actor_adam=[1000]*4,imported_probe_adam=j['probe']['actual_adam_steps_per_rank'])
    once(b/'preparation/resume-joint-receipt.json',result);validate_imported(b,c);return result


def prepare(source,base,staged_repo,commit,authorization):
    """Remote only: code publication plus CP50 state references, no Mac payloads."""
    from .i1_i6_joint_resume import source_manifest
    source,base,repo=map(Path,(source,base,staged_repo));old=read(source/'preparation/launch-config.json');stopped(source)
    proof=source/'preparation/ab-joint-0050-independent-full-sha-cpu.json';q=read(proof)
    if q['status']!='AB_JOINT_FULL_SHA_CPU_PASS':raise ValueError('CP50 qualification incomplete')
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
    c=rebase(old);c.update(base=str(base),execution_id=uuid.uuid4().hex,source_commit=commit,science_source_commit=read(source/'outers/outer-000050/complete.json')['actor']['binding']['source_commit'],prefix_policy_tokens=20,hidden_capture_contract='sample21_keep20_postblock27_native_residual_sum_v1',b_progressive=dict(mode=MODE,resume_outer=50,authorization=authorization),actor_global_micro_batch_size=4)
    # Remove already-consumed first-outer imports; this execution resumes a joint.
    for key in ('startup_sampling_resume','resume_import','resume_pending_probe','node_migration','source_compatibility','source_compatibility_sha256'):c.pop(key,None)
    c['resume_joint']=dict(adapter='i1_i6_joint_v1',source_base=str(source),completed_outer=50,source_commit=old['source_commit'],source_execution_id=old['execution_id'],qualification_proof=str(proof),qualification_proof_sha256=sha(proof),source_joint_sha256=sha(source/'outers/outer-000050/complete.json'))
    parent=source_manifest(source/'repo',old['source_commit']);target=source_manifest(base/'repo',commit,science_source_commit=c['science_source_commit'])
    compat=dict(mode=MODE,science_source_commit=c['science_source_commit'],runtime_source_commit=commit,parent_publication=parent,target_publication=target,controller_changes={k:dict(parent=parent['files_sha256'].get(k),target=target['files_sha256'].get(k))for k in set(parent['files_sha256'])|set(target['files_sha256'])if parent['files_sha256'].get(k)!=target['files_sha256'].get(k)},authorized_candidate_delta=None,authorized_infrastructure_delta=None)
    path=base/'preparation/b-progressive-source-compatibility.json';once(path,compat);c.update(source_compatibility=str(path),source_compatibility_sha256=sha(path));once(base/'preparation/launch-config.json',c)
    verify(c);result=import_complete(base)
    return dict(status='B_PROGRESSIVE_CP50_PREPARED',base=str(base),run_id=c['run_id'],commit=commit,files=len(files),config_sha256=sha(base/'preparation/launch-config.json'),imported_boundary=result['completed_outer'])
