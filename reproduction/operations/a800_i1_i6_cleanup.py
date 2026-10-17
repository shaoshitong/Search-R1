"""Two-phase deletion of explicitly identified, stopped, incorrect A800 runs.

Only the eight run-relative output scopes below can be removed.  Code,
preparation, shared inputs and protected warmup assets are never selected.
Plans contain an exact lstat inventory; execution requires its SHA, rechecks
the inventory and process births, then uses directory FDs without following
symlinks.  No SSH, subprocess, model loading or training is performed here.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time

RUN = '387d3f6e228a488f90c418e2ba47e28e'
OLD_RUN = '411c44aad7ba4e8c90f5fbec9e5ecbaa'
CURRENT_EXECUTION = '1bf0183c24c84abfa8a65c587089145d'
CURRENT_SOURCE = '0de9565c9b1c735635744f7e80a2526a2852e568'
PREFIX = '/data/shaoshitong/search-g1-prefix-selection200-a800-prefix5-'
SCOPES = ('actor', 'probe', 'outers', 'logs', 'validation-retries',
          'final-evaluation', 'checkpoints', 'rollouts')
DENIED_NAMES = {'repo', 'preparation', 'data', 'artifacts', 'sharedasset',
                'sharedassets', 'retrieval', 'retrieval-cache', 'faiss',
                'env', 'venv', '.venv'}
WARMUP = Path(PREFIX + RUN) / 'preparation/warmup-source-epoch-0005.pt'
WARMUP_SHA = 'b5c81ae76c12f677a0663d6e6b0f0482ca63fde0507c2a018aa6034282105594'
PROTECTED_KEYS = ('actor_model_dir', 'official_model_dir', 'model_dir',
    'encoder_model_dir', 'environment_root', 'canonical_train',
    'common_manifest_path', 'composite_manifest_path', 'final_questions',
    'validation_questions', 'fixed_test_features', 'fixed_test_identity_path',
    'source_question_pool', 'grounding_manifest', 'retrieval_index',
    'retrieval_corpus', 'retrieval_encoder', 'retriever_gpu_lib',
    'retrieval_data_cache', 'probe_source_checkpoint', 'probe_source_cpu_proof')
META_NAMES = {'latest.json', 'complete.json', 'initialization-complete.json',
    'metrics.json', 'probe-frozen.json', 'pipeline-status.json',
    'cleanup-status.json', 'phase-timings.json', 'attempt.json',
    'diagnostic-complete.json', 'storage-warning.json'}
META_KEYS = {'run_id', 'execution_id', 'source_commit', 'status', 'outer',
    'completed_outer', 'complete_outer', 'complete_outers', 'actual_adam_steps_per_rank',
    'new_adam_steps_per_rank', 'scheduler_completed_outer', 'observed_unix',
    'checked_unix', 'sha256', 'bytes', 'binding_sha256', 'qa_count', 'em', 'f1',
    'actual_learning_rate', 'new_adam_steps_per_rank', 'last_update_outer',
    'periodic_probe_update', 'cleaned', 'owned_births_all_gone',
    'remaining_owned_births', 'exit_code', 'success', 'error_class', 'error_type'}
MAX_JSON = 64 * 1024 * 1024


def _sha(path):
    h = hashlib.sha256()
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _read(path):
    p = Path(path)
    if p.is_symlink() or p.stat().st_size > MAX_JSON:
        raise ValueError('Require a bounded real JSON file: ' + str(p))
    fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as f:
        return json.load(f)


def _once(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'w') as f:
        json.dump(value, f, sort_keys=True, allow_nan=False); f.write('\n')
        f.flush(); os.fsync(f.fileno())
    return dict(path=str(path), bytes=path.stat().st_size, sha256=_sha(path))


def _canonical(path, *, directory=False):
    p = Path(path)
    if (not p.is_absolute() or '..' in p.parts or p.is_symlink()
            or p.resolve(strict=True) != p
            or (directory and not p.is_dir())):
        raise ValueError('Require a canonical non-symlink path: ' + str(p))
    return p


def _base(handle, owner):
    run = handle['run_id']; base = _canonical(handle['base'], directory=True)
    if run not in (RUN, OLD_RUN):
        raise ValueError('Deletion authorization does not include another experiment')
    fresh = PREFIX + run
    if str(base) not in (fresh, fresh + '-resume-' + handle['execution_id']):
        raise ValueError('Base does not match the exact explicitly supplied run/execution')
    if not re.fullmatch('[0-9a-f]{32}', handle['execution_id']) or not re.fullmatch('[0-9a-f]{40}', handle['source_commit']):
        raise ValueError('Actual deployment identity is incomplete')
    deployment = _canonical(base / 'preparation/deployment.json')
    actual = _read(deployment)
    for key in ('base', 'run_id', 'execution_id', 'source_commit'):
        if actual.get(key) != handle[key]:
            raise ValueError('Actual remote deployment differs from supplied handle: ' + key)
    if (base.stat().st_uid != owner or deployment.stat().st_uid != owner
            or (base/'preparation').stat().st_uid != owner
            or any('owner_uid' in obj and obj['owner_uid'] != owner for obj in (handle, actual))):
        raise ValueError('Actual namespace/deployment ownership differs')
    return base, dict(path=str(deployment), sha256=_sha(deployment), owner_uid=owner,
                      base_dev=base.stat().st_dev, base_inode=base.stat().st_ino)


def _signature(st):
    kind = ('directory' if stat.S_ISDIR(st.st_mode) else 'file' if stat.S_ISREG(st.st_mode)
            else 'symlink' if stat.S_ISLNK(st.st_mode) else 'unsupported')
    return dict(kind=kind, dev=st.st_dev, inode=st.st_ino, mode=st.st_mode,
                uid=st.st_uid, bytes=st.st_size, mtime_ns=st.st_mtime_ns)


def _scan_fd(fd, rel, owner, device, result):
    st = os.fstat(fd)
    if st.st_uid != owner or st.st_dev != device:
        raise ValueError('Foreign owner or mounted filesystem inside deletion scope: ' + rel)
    result.append(dict(relative=rel, **_signature(st)))
    for name in sorted(os.listdir(fd)):
        s = os.stat(name, dir_fd=fd, follow_symlinks=False)
        item = rel + '/' + name
        if s.st_uid != owner or s.st_dev != device:
            raise ValueError('Foreign owner or filesystem inside deletion scope: ' + item)
        signature = _signature(s)
        if signature['kind'] == 'directory':
            diagnostic_config = bool(re.fullmatch(r'outers/outer-\d{6}/diagnostic/preparation', item))
            if name.lower() in DENIED_NAMES and not diagnostic_config:
                raise ValueError('Protected directory name inside output scope: ' + item)
            child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            try:
                if os.fstat(child).st_ino != s.st_ino:
                    raise ValueError('Directory changed while planning')
                if diagnostic_config:
                    # This is generated per-outer diagnostic configuration,
                    # not the experiment's protected top-level preparation.
                    # Only its exact native single config is in the allowlist.
                    directory = Path(os.readlink('/proc/self/fd/%d' % child))
                    if sorted(os.listdir(child)) != ['launch-config.json']:
                        raise ValueError('Unexpected asset in generated diagnostic preparation')
                    config = _read(directory/'launch-config.json')
                    if (config.get('run_id') not in (RUN, OLD_RUN) or
                            config.get('diagnostic_only') is not True or
                            config.get('base') != str(directory.parent)):
                        raise ValueError('Diagnostic configuration is not the exact owned output')
                _scan_fd(child, item, owner, device, result)
            finally:
                os.close(child)
        elif signature['kind'] in ('file', 'symlink'):
            # A nested link may be unlinked, but neither its target nor children
            # are visited.  A top-level scope link is rejected below.
            result.append(dict(relative=item, **signature))
        else:
            raise ValueError('Special file in deletion scope: ' + item)


def _inventory(base, owner):
    result, absent = [], []
    base = Path(base)
    fd = os.open(base, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for scope in SCOPES:
            try:
                s = os.stat(scope, dir_fd=fd, follow_symlinks=False)
            except FileNotFoundError:
                absent.append(scope); continue
            if not stat.S_ISDIR(s.st_mode):
                raise ValueError('Top-level deletion scope must be a real directory: ' + scope)
            child = os.open(scope, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            try:
                _scan_fd(child, scope, owner, os.fstat(fd).st_dev, result)
            finally:
                os.close(child)
    finally:
        os.close(fd)
    return result, absent


def _born(pid, ticks):
    try:
        row = Path('/proc/%d/stat' % pid).read_text().rsplit(')', 1)[1].split()
        return int(row[19]) == int(ticks)
    except (FileNotFoundError, ProcessLookupError):
        return False


def _identities(value):
    if isinstance(value, dict):
        if type(value.get('pid')) is int and type(value.get('start_ticks')) is int:
            yield value['pid'], value['start_ticks']
        for item in value.values():
            yield from _identities(item)
    elif isinstance(value, list):
        for item in value:
            yield from _identities(item)


def _dead(base):
    """Use real process births and a bounded /proc identity scan, not lockfiles."""
    identities = set()
    files = [base/'logs/process-ledger.json', base/'preparation/submit-receipt.json']
    files += list((base/'preparation').glob('*launch*.json'))
    for path in files:
        if path.is_file() and not path.is_symlink():
            identities.update(_identities(_read(path)))
    live = [dict(pid=pid, start_ticks=ticks) for pid,ticks in identities if _born(pid,ticks)]
    bound = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            args = (proc/'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            cwd = os.readlink(proc/'cwd')
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        pattern = re.escape(str(base)) + r'(?=/|\s|$)'
        if re.search(pattern, args) or re.search(pattern, cwd):
            bound.append(int(proc.name))
    if live or bound:
        raise ValueError('Target namespace still has actual writers/processes: ' + str(dict(live=live,bound=bound)))
    return dict(recorded_births_checked=len(identities), matching_births_present=0,
                actual_base_bound_processes_present=0, checked_unix=time.time())


def _terminal(path, expected_sha):
    path = _canonical(path)
    if _sha(path) != expected_sha:
        raise ValueError('Selected actual terminal proof changed')
    value = _read(path)
    if isinstance(value.get('stdout'), str):
        value = json.loads(value['stdout'])
    if (value.get('terminal') is not True or value.get('run_id') != RUN
            or value.get('execution_id') != CURRENT_EXECUTION
            or value.get('source_commit') != CURRENT_SOURCE
            or value.get('pipeline_alive') is not False
            or value.get('remaining_owned_births') or value.get('remaining_base_bound_processes')
            or not value.get('manual_stop_not_failure')
            or not isinstance(value.get('protected_asset_config'), dict)):
        raise ValueError('Require the exact actual human-stop terminal/protected-asset proof')
    cleanup = value.get('cleanup') or {}
    if 'value' in cleanup:
        cleanup = cleanup['value']
    if cleanup.get('owned_births_all_gone') is not True or cleanup.get('cleaned') is not True:
        raise ValueError('All task-owned processes must really have released')
    return path, value


def _outside_outputs(path, bases):
    p = Path(path)
    for base in bases:
        for scope in SCOPES:
            selected = base/scope
            if p == selected or p.is_relative_to(selected) or selected.is_relative_to(p):
                raise ValueError('A protected/output path overlaps a deletion scope: ' + str(p))


def _output_path(path, bases):
    p=Path(path)
    if (not p.is_absolute() or '..' in p.parts or not str(p).startswith('/data/shaoshitong/')):
        raise ValueError('Keep cleanup plans/receipts in explicit canonical remote audit storage')
    _canonical(p.parent,directory=True);_outside_outputs(p,bases)
    return p


def _protected(config, bases):
    paths = {str(WARMUP), '/data/search-g1/assets', '/data/search-r1/assets',
        '/data/search-g1/env',
        '/data/shaoshitong/Search-G1-a800-fresh200-7895330a2c584bba9940720e98f006ef'}
    for key in PROTECTED_KEYS:
        value = config.get(key)
        if isinstance(value, str) and value.startswith('/data/'):
            paths.add(value)
    result = []
    for name in sorted(paths):
        p = _canonical(name); _outside_outputs(p, bases); st=p.stat()
        result.append(dict(path=str(p), **_signature(st)))
    return result


def _pin(src, dst, expected_sha):
    src = _canonical(src); dst = Path(dst)
    if _sha(src) != expected_sha:
        raise ValueError('Asset full SHA differs before protection: ' + str(src))
    if dst.exists():
        if dst.is_symlink() or dst.stat().st_ino != src.stat().st_ino or dst.stat().st_dev != src.stat().st_dev:
            raise ValueError('Existing protected pin belongs to another actual asset')
    else:
        os.link(src, dst, follow_symlinks=False)
    if (dst.stat().st_dev,dst.stat().st_ino)!=(src.stat().st_dev,src.stat().st_ino):
        raise ValueError('Protected hardlink does not retain the full-SHA verified original inode')
    return dict(source=str(src), path=str(dst), bytes=dst.stat().st_size,
                dev=dst.stat().st_dev, inode=dst.stat().st_ino, sha256=expected_sha)


def _audit_metadata(base, inventory):
    records = []
    for row in inventory:
        p = base/row['relative']
        if row['kind'] != 'file' or p.name not in META_NAMES or row['bytes'] > 1024*1024:
            continue
        value = _read(p)
        if not isinstance(value,dict):
            continue
        projection = {k:v for k,v in value.items() if k in META_KEYS
                      and (v is None or isinstance(v,(str,int,float,bool,list)))}
        for key in ('actor','probe'):
            if isinstance(value.get(key),dict):
                projection[key]={k:v for k,v in value[key].items() if k in META_KEYS
                    and (v is None or isinstance(v,(str,int,float,bool,list)))}
        records.append(dict(relative=row['relative'],bytes=row['bytes'],sha256=_sha(p),scalar=projection))
    return records


def _inputs(args):
    handles=[]
    if args.targets:
        value = _read(_canonical(args.targets))
        handles.extend(value['targets'] if isinstance(value,dict) else value)
    for path in args.handle or []:
        handles.append(_read(_canonical(path)))
    if not handles or not all(isinstance(h,dict) for h in handles):
        raise ValueError('Supply actual explicit deployment handles; discovery is not authorized')
    if len({h['base'] for h in handles}) != len(handles):
        raise ValueError('Duplicate target namespace')
    return handles


def plan(args):
    handles = _inputs(args); owner = args.expected_owner_uid
    if owner != os.getuid():
        raise ValueError('Delete only output files owned by the actual invoking task UID')
    if os.path.lexists(args.output):
        raise ValueError('Plan output already exists; inspect it rather than replaying preparation')
    bound = [(h,*_base(h,owner)) for h in handles]; bases=[x[1] for x in bound]
    _output_path(args.output,bases)
    terminal_path, terminal = _terminal(args.terminal_proof,args.terminal_proof_sha256)
    asset_config=terminal['protected_asset_config']
    if (asset_config.get('probe_source_checkpoint')!=str(WARMUP)
            or asset_config.get('probe_source_sha256')!=WARMUP_SHA):
        raise ValueError('Actual retained warmup source identity differs from mandatory protected asset')
    protected_dir=Path(args.protected_assets)
    if (not protected_dir.is_absolute() or not str(protected_dir).startswith('/data/shaoshitong/')
            or '..' in protected_dir.parts):
        raise ValueError('Pin retained assets in explicit canonical remote protected storage')
    _outside_outputs(protected_dir,bases)
    ancestor=protected_dir
    while not os.path.lexists(ancestor):
        ancestor=ancestor.parent
    _canonical(ancestor,directory=True)
    protected_dir.mkdir(parents=True,exist_ok=True);_canonical(protected_dir,directory=True)
    if protected_dir.stat().st_uid != owner:
        raise ValueError('Protected asset directory owner differs')
    assets = _protected(terminal['protected_asset_config'],bases)
    warmup = _pin(WARMUP,protected_dir/('warmup-epoch0005-'+WARMUP_SHA+'.pt'),WARMUP_SHA)
    terminal_pin = _pin(terminal_path,protected_dir/('terminal-'+args.terminal_proof_sha256+'.json'),args.terminal_proof_sha256)
    targets=[]
    for handle,base,deployment in bound:
        dead=_dead(base); inventory,absent=_inventory(base,owner)
        audit=_once(protected_dir/('audit-'+handle['execution_id']+'-'+str(time.time_ns())+'.json'),
            dict(run_id=handle['run_id'],execution_id=handle['execution_id'],source_commit=handle['source_commit'],
                 source_base=str(base),deployment=deployment,terminal_proof_sha256=args.terminal_proof_sha256,
                 preserved_before_delete=True,metadata=_audit_metadata(base,inventory)))
        targets.append(dict(identity={k:handle[k] for k in ('base','run_id','execution_id','source_commit')},
            deployment=deployment,actual_process_exit=dead,inventory=inventory,absent_scopes=absent,
            preserved_scalar_audit=audit,regular_file_bytes=sum(x['bytes'] for x in inventory if x['kind']=='file')))
    result=dict(schema_version=1,status='PLANNED_NOT_DELETED',created_unix=time.time(),
        expected_owner_uid=owner,scopes=list(SCOPES),explicit_targets=targets,protected_paths=assets,
        warmup_pin=warmup,terminal_pin=terminal_pin,terminal_proof_sha256=args.terminal_proof_sha256,
        helper_source_sha256=_sha(__file__),no_raw_QA_or_rollout_payload_copied=True,
        shared_assets_code_preparation_kept=True,symlink_targets_never_followed=True)
    proof=_once(args.output,result)
    return dict(status=result['status'],plan=proof,target_count=len(targets),
                regular_file_bytes=sum(x['regular_file_bytes'] for x in targets),warmup_pin=warmup)


def _remove_fd(parent,name,relative,expected,owner,device,counts):
    row=expected[relative]; actual=_signature(os.stat(name,dir_fd=parent,follow_symlinks=False))
    if actual != {k:row[k] for k in actual} or actual['uid']!=owner or actual['dev']!=device:
        raise ValueError('A deletion entry changed after accepted plan: '+relative)
    if row['kind']=='directory':
        fd=os.open(name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=parent)
        try:
            if os.fstat(fd).st_ino!=row['inode']:
                raise ValueError('Directory raced before deletion')
            for child in sorted(os.listdir(fd)):
                key=relative+'/'+child
                if key not in expected:
                    raise ValueError('Unplanned child appeared; preserve and reconcile: '+key)
                _remove_fd(fd,child,key,expected,owner,device,counts)
        finally:
            os.close(fd)
        os.rmdir(name,dir_fd=parent);counts['directories']+=1
    else:
        # unlinkat does not traverse a file/symlink target. Directory traversal
        # always uses O_NOFOLLOW above, including during races.
        os.unlink(name,dir_fd=parent);counts[row['kind']+'s']+=1


def execute(args):
    plan_path=_canonical(args.plan)
    if _sha(plan_path)!=args.plan_sha256:
        raise ValueError('Execute only the exact reviewed plan SHA')
    value=_read(plan_path)
    if (value.get('status')!='PLANNED_NOT_DELETED' or value.get('schema_version')!=1
            or value.get('scopes')!=list(SCOPES) or value.get('helper_source_sha256')!=_sha(__file__)):
        raise ValueError('Plan scope/source differs from this exact cleanup implementation')
    output=Path(args.output);intent=output.with_name(output.name+'.execute-intent.json')
    claim=plan_path.with_name(plan_path.name+'.execute-claim.json')
    if output.exists() or intent.exists() or claim.exists():
        raise ValueError('Cleanup onceclaim exists; inspect actual receipt, never replay blindly')
    owner=value['expected_owner_uid'];targets=value['explicit_targets'];bases=[]
    if owner != os.getuid():
        raise ValueError('Cleanup actual UID differs from the planned owner')
    # Revalidate every namespace and every inventory before the first unlink.
    for target in targets:
        base,deployment=_base(target['identity'],owner);bases.append(base)
        if deployment!=target['deployment']:
            raise ValueError('Actual deployment changed after cleanup plan')
        _dead(base)
        inventory,absent=_inventory(base,owner)
        if inventory!=target['inventory'] or absent!=target['absent_scopes']:
            raise ValueError('Output tree changed after plan; no deletion performed')
        if _sha(target['preserved_scalar_audit']['path'])!=target['preserved_scalar_audit']['sha256']:
            raise ValueError('Necessary small audit must remain independently preserved')
    _output_path(output,bases);_outside_outputs(plan_path,bases)
    _terminal(value['terminal_pin']['path'],value['terminal_proof_sha256'])
    for protected in value['protected_paths']:
        path=_canonical(protected['path']);_outside_outputs(path,bases)
        if _signature(path.stat())!= {k:protected[k] for k in _signature(path.stat())}:
            raise ValueError('Protected input metadata changed since planning')
    pin=value['warmup_pin'];warmup=_canonical(pin['path'])
    if (_sha(warmup)!=WARMUP_SHA or (warmup.stat().st_dev,warmup.stat().st_ino)!=(pin['dev'],pin['inode'])
            or (WARMUP.stat().st_dev,WARMUP.stat().st_ino)!=(pin['dev'],pin['inode'])):
        raise ValueError('Retained warmup full SHA must still pass before any deletion')
    _once(claim,dict(plan_sha256=args.plan_sha256,receipt=str(output),started_unix=time.time(),pid=os.getpid()))
    _once(intent,dict(plan=str(plan_path),plan_sha256=args.plan_sha256,started_unix=time.time(),pid=os.getpid()))
    outcomes=[]
    try:
        for target,base in zip(targets,bases):
            _dead(base)
            expected={r['relative']:r for r in target['inventory']}
            counts=dict(files=0,symlinks=0,directories=0)
            fd=os.open(base,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
            try:
                for scope in SCOPES:
                    if scope in expected:
                        _remove_fd(fd,scope,scope,expected,owner,os.fstat(fd).st_dev,counts)
            finally:
                os.close(fd)
            outcomes.append(dict(identity=target['identity'],removed=counts,
                absent_all_selected_scopes=all(not os.path.lexists(base/scope) for scope in SCOPES),
                code_preparation_preserved=(base/'repo').is_dir() and (base/'preparation/deployment.json').is_file()))
        if (_sha(warmup)!=WARMUP_SHA
                or (WARMUP.stat().st_dev,WARMUP.stat().st_ino)!=(pin['dev'],pin['inode'])):
            raise ValueError('Protected warmup post-cleanup SHA changed')
        for protected in value['protected_paths']:
            path=_canonical(protected['path'])
            if _signature(path.stat())!={k:protected[k] for k in _signature(path.stat())}:
                raise ValueError('Protected input metadata changed during cleanup')
        result=dict(status='EXPLICIT_WRONG_RUN_OUTPUTS_DELETED_PROTECTED_ASSETS_RETAINED',
            plan_sha256=args.plan_sha256,finished_unix=time.time(),outcomes=outcomes,
            warmup_sha256=WARMUP_SHA,warmup_pin=str(warmup),protected_inputs_retained=True,
            source_code_preparation_shared_assets_not_selected=True,symlink_targets_never_followed=True)
    except BaseException as error:
        _once(output,dict(status='CLEANUP_PARTIAL_OR_FAILED_RECONCILE_REQUIRED',plan_sha256=args.plan_sha256,
            outcomes=outcomes,error_class=type(error).__name__,error=str(error),finished_unix=time.time()))
        raise
    return dict(result,receipt=_once(output,result))


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='action',required=True)
    a=s.add_parser('plan');a.add_argument('--targets');a.add_argument('--handle',action='append')
    a.add_argument('--terminal-proof',required=True);a.add_argument('--terminal-proof-sha256',required=True)
    a.add_argument('--protected-assets',required=True);a.add_argument('--expected-owner-uid',type=int,default=os.getuid())
    a.add_argument('--output',required=True)
    b=s.add_parser('execute');b.add_argument('--plan',required=True);b.add_argument('--plan-sha256',required=True);b.add_argument('--output',required=True)
    args=p.parse_args();print(json.dumps(plan(args) if args.action=='plan' else execute(args),allow_nan=False))


if __name__=='__main__':
    main()
