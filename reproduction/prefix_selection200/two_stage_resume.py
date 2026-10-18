"""Exact stopped progressive-joint import for the human-authorized two-stage suffix."""
import json,os
from pathlib import Path
from .progressive_resume import read,sha,writers_gone
from .two_stage_selection import validate,MODE as SELECTION_MODE

PROTECTED=('actor.py','actor_safety.py','interaction_protocol.py','grounding.py','prefix5_model.py','probe_plan.py','probe_epoch_checkpoint.py')

def verify(config):
    from .progressive_resume import _verify_progressive
    proof=_verify_progressive(config);spec=validate(config)
    auth=config['two_stage_authorization'];intent=read(auth)
    if sha(auth)!=config['two_stage_authorization_sha256'] or intent['selection']!=spec or intent['resume_outer']!=config['resume_joint']['completed_outer']:
        raise ValueError('Two-stage delta requires its exact latest explicit human authorization')
    parent=proof['parent_publication'];target=proof['target_publication']
    names=['reproduction/prefix_selection200/'+n for n in PROTECTED]+['reproduction/prefix20_group_joint100/loss.py']
    if any(parent['files_sha256'][n]!=target['files_sha256'][n] for n in names):
        raise ValueError('Two-stage selection may not change actor/probe loss/optimizer/interaction math')
    critical=set(parent['critical_ast_sha256'])-{'score'}
    if any(parent['critical_ast_sha256'][n]!=target['critical_ast_sha256'].get(n) for n in critical):
        raise ValueError('Existing probe model/loss/training/Adam/restore AST changed')
    if proof['authorized_two_stage_selection']!=spec or proof['human_intent_sha256']!=config['two_stage_authorization_sha256']:
        raise ValueError('Published source envelope lost the two-stage delta')
    return proof

def plan(base,config=None):
    base=Path(base);c=config or read(base/'preparation/launch-config.json');proof=verify(c)
    spec=c['resume_joint'];n=spec['completed_outer'];source=Path(spec['source_base']);original=read(source/'preparation/launch-config.json')
    if base==source or base.parent!=source.parent or c['run_id']!=original['run_id'] or c['execution_id']==original['execution_id']:
        raise ValueError('Keep run identity in an isolated new execution')
    if c['two_stage_selection']['effective_from_outer']!=n+1 or n not in (60,61):
        raise ValueError('The latest human authorized resuming complete60 or61')
    if sha(source/'preparation/launch-config.json')!=spec['source_config_sha256'] or original['source_commit']!=spec['source_commit'] or original['execution_id']!=spec['source_execution_id']:
        raise ValueError('Stopped source config/commit/execution changed')
    writers_gone(source,c)
    joint_path=source/'outers'/('outer-%06d'%n)/'complete.json';j=read(joint_path);qualified=read(spec['qualification_proof'])
    if (sha(joint_path)!=spec['source_joint_sha256'] or sha(spec['qualification_proof'])!=spec['qualification_proof_sha256']
        or qualified['status']!='STOPPED_PROGRESSIVE_JOINT_FULL_SHA_CPU_READBACK_PASS' or qualified['outer']!=n
        or qualified['source_joint_sha256']!=sha(joint_path) or not qualified['actual_cpu_deserialization']
        or not qualified['training_stopped'] or not qualified['stopped_writer_recovery_qualification']
        or qualified['actor_actual_adam_steps_per_rank']!=j['actor']['actual_adam_steps_per_rank']
        or qualified['probe']['actual_adam_steps_per_rank']!=j['probe']['actual_adam_steps_per_rank']):
        raise ValueError('Resume requires genuine stopped whole-joint SHA/CPU evidence, not probe-only qualification')
    for row in qualified['verified_files']:
        p=Path(row['path']);s=p.stat()
        if (s.st_ino,s.st_size,s.st_mtime_ns,str(os.statvfs(p).f_fsid))!=(row['inode'],row['bytes'],row['mtime_ns'],str(row['shared_filesystem_fsid'])):
            raise ValueError('Already CPU-qualified immutable source changed: '+str(p))
    return dict(status='QUALIFIED_STOPPED_TWO_STAGE_IMPORT_READY',adapter='i1_i6_joint_v1',
        run_id=c['run_id'],execution_id=c['execution_id'],completed_outer=n,next_outer=n+1,source_base=str(source),
        source_joint_sha256=sha(joint_path),qualification_proof_sha256=sha(spec['qualification_proof']),
        actor_actual_adam=j['actor']['actual_adam_steps_per_rank'],probe_actual_adam=j['probe']['actual_adam_steps_per_rank'],
        phase_reuse={},sealed_files=len(qualified['verified_files']),full_payload_sha_repeated=False,
        source_compatibility_sha256=c['source_compatibility_sha256'],selection=validate(c))

def import_complete(base):
    base=Path(base);c=read(base/'preparation/launch-config.json');result=plan(base,c)
    receipt=base/'preparation/resume-joint-receipt.json'
    if receipt.exists():validate_imported_boundary(base,c);return read(receipt)
    intent=base/'preparation/two-stage-import-intent.json'
    if intent.exists():raise RuntimeError('Reconcile interrupted import; never repeat over a half-import')
    with intent.open('x') as f:json.dump(result,f,sort_keys=True)
    source=Path(result['source_base']);n=result['completed_outer'];j=read(source/'outers'/('outer-%06d'%n)/'complete.json')
    def cp(src,dst):
        dst.parent.mkdir(parents=True,exist_ok=True)
        with dst.open('xb') as f:f.write(Path(src).read_bytes())
    for i in range(1,n+1):
        root=base/'outers'/('outer-%06d'%i);prior=source/'outers'/root.name
        for name in ('candidate-budget.json','pool-state-at-completion.json','complete.json'):
            if (prior/name).exists():cp(prior/name,root/name)
    cp(j['pool_snapshot'],base/'question-pool/state.json');cp(source/'question-pool/binding.json',base/'question-pool/binding.json')
    (base/'actor').mkdir(exist_ok=True);(base/'probe').mkdir(exist_ok=True)
    with (base/'actor/latest.json').open('x') as f:json.dump(j['actor'],f)
    with (base/'probe/latest.json').open('x') as f:json.dump(j['probe'],f)
    os.link(j['probe_snapshot'],base/'probe/latest.pt')
    cp(source/'latest-joint.json',base/'latest-joint.json')
    result.update(status='I1_I6_FULL_JOINT_IMPORTED',actor_updates_reexecuted=0,probe_updates_reexecuted=0,discarded_uncompleted_outer=n+1)
    with receipt.open('x') as f:json.dump(result,f,sort_keys=True)
    validate_imported_boundary(base,c);return result

def validate_imported_boundary(base,c):
    base=Path(base);current=plan(base,c);receipt=read(base/'preparation/resume-joint-receipt.json');n=current['completed_outer']
    if receipt['status']!='I1_I6_FULL_JOINT_IMPORTED' or receipt['source_joint_sha256']!=current['source_joint_sha256']:
        raise ValueError('Imported actual joint provenance changed')
    j=read(Path(current['source_base'])/'latest-joint.json')
    if read(base/'latest-joint.json')!=j or read(base/'actor/latest.json')!=j['actor'] or read(base/'probe/latest.json')!=j['probe']:
        raise ValueError('Imported actor/probe/cursor cannot be substituted or reinitialized')
    s=Path(j['probe_snapshot']).stat();t=(base/'probe/latest.pt').stat()
    if (s.st_dev,s.st_ino)!=(t.st_dev,t.st_ino):raise ValueError('Imported probe alias changed')
    if sha(base/'question-pool/state.json')!=j['pool_sha256']:raise ValueError('Imported data position changed')
    return n
