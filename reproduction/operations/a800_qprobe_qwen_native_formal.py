"""Published, one-shot native-Qwen formal200 operations after actual warmup10.

Stage/prepare/bank-prepare are CPU-only. The configured exact warmup completion
and release audit gate bank-encode; full bank SHA and encoder exit gate binding.
A context-free release/deployment review gates one fresh actor launch. There is
no retry, restore, foreign stop, or active-source edit action.
"""
import argparse
import ast
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid

sys.dont_write_bytecode=True
ROOT=Path('/Users/shaoshitong/Documents/Search-G1')
OPS=Path(__file__).resolve().parents[2]
RUNTIME=ROOT/'qprobe-qwen-native-driver-process-fix'
SUPPORT_PATH='reproduction/operations/a800_qprobe_qwen_block_warmup10.py'
SUPPORT_COMMIT='a4d2328574c3a51dd3f201d7ffcad53acd8fd839'
SUPPORT_SHA='f95f3314a58b5499544f050c3e886fd711ca5a39e34c9dac631757c7c75bc635'
ROLE='a800_qwen_native_formal_prospective_intent'
PREFIX='/data/shaoshitong/Search-G1-a800-fresh200-'
HELPER='reproduction.a800_common.qprobe_qwen_formal_ops'
QUALIFIED_WARMUP_BASE='/data/shaoshitong/Search-G1-qprobe-qwen-lastblock-warmup10-096c8297cba74f8089063e763686671e'
QUALIFIED_WARMUP_SOURCE='e81445905bf8b15dbb0bdec3afa29d99ab0b11d8'
QUALIFIED_WARMUP_MAP='006912cc7817b2ae8abde32a85c42d21b2c1976bf0b5618066c6aa9d2f0674ad'


def support():
    raw=(OPS/SUPPORT_PATH).read_bytes();assert hashlib.sha256(raw).hexdigest()==SUPPORT_SHA
    assert subprocess.check_output(['git','-C',str(OPS),'show',SUPPORT_COMMIT+':'+SUPPORT_PATH])==raw
    spec=importlib.util.spec_from_file_location('published_native_warmup_transport',OPS/SUPPORT_PATH)
    warmup=importlib.util.module_from_spec(spec);spec.loader.exec_module(warmup)
    scaffold,base=warmup.load_support();return warmup,scaffold,base


def authority(warmup,base,intent):
    human=base.document(warmup.HUMAN);warmup.validate_human(human)
    assert human['formal200_explicitly_authorized_by_latest_goal'] is True and human['warmup_then_continue_formal200'] is True
    assert human['formal_head_epochs_per_complete_outer']==2 and human['formal_replay_completed_outer_window']==8 and human['no_minibatch_or_global_weight_renormalization'] is True
    assert intent['human_intent_sha256']==base.digest(warmup.HUMAN)
    assert intent['role']==ROLE and intent['base']==PREFIX+intent['deployment_id'] and re.fullmatch('[0-9a-f]{32}',intent['deployment_id'])
    assert intent['native_formal_200_authorized'] is True and intent['source_publication_authorized'] is True and intent['target_updates']==200 and intent['head_device']=='cuda:0'
    assert intent['warmup_handle']['base']==intent['warmup_base'] and intent['warmup_handle']['source_commit']==intent['warmup_source_commit']
    assert hashlib.sha256(base.canonical(intent['warmup_handle']['source_sha256'])).hexdigest()==intent['warmup_source_map_sha256']
    assert (intent['warmup_base'],intent['warmup_source_commit'],intent['warmup_source_map_sha256'])==(QUALIFIED_WARMUP_BASE,QUALIFIED_WARMUP_SOURCE,QUALIFIED_WARMUP_MAP)
    if intent.get('qualified_bank_origin') is not None:
        origin=intent['qualified_bank_origin']
        assert origin['role']=='native_qwen_preupdate_qualified_bank_origin_v1' and intent['bank_base']==origin['bank_base']
        assert intent['base']!=origin['base'] and intent['source_commit']!=origin['source_commit']


def stage_code(warmup,scaffold,base):
    code=warmup.program(scaffold,base,'stage')
    old_role=repr(warmup.ROLE);old_prefix=repr(warmup.PREFIX)+"+intent['deployment_id']"
    assert code.count(old_role)==2 and code.count(old_prefix)==2
    code=code.replace(old_role,repr(ROLE)).replace(old_prefix,repr(PREFIX)+"+intent['deployment_id']")
    marker=';seen=set()\nwhile True:';assert code.count(marker)==1
    return code.replace(marker,"\nif intent.get('qualified_bank_origin') is not None:\n origin=intent['qualified_bank_origin'];expected['preparation/qualified-bank-terminal-audit.json']=(origin['terminal_audit_bytes'],origin['terminal_audit_sha256'])\nseen=set()\nwhile True:")


REMOTE=r'''
bind(json.load(sys.stdin));sys.path.insert(0,str(repo))
def inventory():
 queried=time.time();rows=list(csv.reader(io.StringIO(subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=20))))
 apps=list(csv.reader(io.StringIO(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True,timeout=20))))
 devices=[dict(physical_index=int(row[0]),gpu_uuid=row[1].strip(),used_mib=float(row[2]),util_percent=float(row[3])) for row in rows]
 assert len({row['physical_index'] for row in devices})==len(devices) and len({row['gpu_uuid'] for row in devices})==len(devices)
 assert all(row['physical_index']>=0 and math.isfinite(row['used_mib']) and row['used_mib']>=0 and math.isfinite(row['util_percent']) and 0<=row['util_percent']<=100 for row in devices)
 busy={row[0].strip() for row in apps if row};idle=[row['physical_index'] for row in devices if row['used_mib']<=256 and row['util_percent']==0 and row['gpu_uuid'] not in busy]
 return dict(devices=devices,idle_training_indices=[index for index in idle if index not in (5,6)],auxiliary_5_6_idle=all(index in idle for index in (5,6)),queried_unix=queried)
action=p['action'];args=['/data/search-g1/env/bin/python','-m','reproduction.a800_common.qprobe_qwen_formal_ops','--base',str(base),'--action',action]
if action=='warmup-admit':
 path=base/'preparation/warmup-completion-input.json';once(path,p['warmup_receipt']);args+=['--warmup-receipt',str(path)]
if action=='launch':
 path=base/'preparation/native-formal-release-review.json';once(path,p['review']);args+=['--review',str(path)]
if action=='bank-encode':
 index=p['physical_index'];assert type(index) is int and index>=0 and index not in (5,6)
 actual=inventory();assert index in actual['idle_training_indices'];args+=['--physical-index',str(index)]
if action=='observe':
 result=subprocess.run(args,cwd=repo,env=environment(),stdin=subprocess.DEVNULL,capture_output=True,timeout=100)
 assert len(result.stdout)<=1024*1024
 rows=[json.loads(line.split(':',1)[1]) for line in result.stdout.decode().splitlines() if line.startswith('NATIVE_FORMAL_JSON:')]
 assert result.returncode==0 and len(rows)==1;value=rows[0]
 if p.get('admission'):value['actual_idle_admission']=inventory()
 emit(value)
else:
 assert not (base/'logs'/(action+'-launch.json')).exists()
 once(base/'logs'/('native-formal-operator-'+action+'-launch-once.json'),dict(action=action,source_commit=intent['source_commit'],claimed_unix=time.time(),no_automatic_retry=True))
 with (base/'logs'/(action+'-worker.log')).open('xb') as log:
  child=subprocess.Popen(['/data/search-g1/env/bin/python','-c',p['worker'],json.dumps(dict(base=str(base),repo=str(repo),action=action,args=args))],cwd=repo,env=environment(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  record=dict(action=action,source_commit=intent['source_commit'],process=birth(child.pid),no_automatic_retry=True,launched_unix=time.time())
  once(base/'logs'/(action+'-launch.json'),record)
 emit(record)
'''


def remote_code(base):
    common=base.COMMON
    role="'a800_token_binary_warmup10_only_intent'";prefix="'/data/shaoshitong/Search-G1-qprobe-token-binary-warmup10-'+intent['deployment_id']"
    assert common.count(role)==common.count(prefix)==1
    return common.replace(role,repr(ROLE)).replace(prefix,repr(PREFIX)+"+intent['deployment_id']")+'\n'+REMOTE


def qualified_bank_origin(base,args,repo,mapping):
    supplied=(args.qualified_bank_origin_handle,args.bank_admitted_receipt,args.terminal_audit_receipt)
    if not any(path is not None for path in supplied):return None
    assert all(path is not None for path in supplied), 'All three qualified bank-origin inputs are required'
    path=repo/'reproduction/a800_common/qprobe_qwen_formal_ops.py'
    spec=importlib.util.spec_from_file_location('published_native_bank_origin_binding',path)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    old=base.document(args.qualified_bank_origin_handle);receipt=base.document(args.bank_admitted_receipt)
    assert old['base']==helper.QUALIFIED_BANK_ORIGIN and old['source_commit']==helper.QUALIFIED_BANK_SOURCE
    assert hashlib.sha256(base.canonical(old['source_sha256'])).hexdigest()==helper.QUALIFIED_BANK_MAP
    assert receipt['exit_code']==0 and receipt['base']==old['base'] and receipt['source_commit']==old['source_commit']
    observation=receipt['observation'];assert observation['base']==old['base'] and observation['source_commit']==old['source_commit']
    item=observation['files']['logs/native-bank-admitted.json'];complete=observation['files'][old['bank_base']+'/complete.json']
    assert item['sha256']==helper.QUALIFIED_BANK_ADMIT and complete['sha256']==helper.QUALIFIED_BANK_COMPLETE
    assert complete['metadata']['source_commit']==old['source_commit'] and complete['metadata']['feature_manifest_sha256']==helper.QUALIFIED_BANK_MANIFEST
    audit_path=args.terminal_audit_receipt.resolve();assert base.digest(audit_path)==helper.QUALIFIED_BANK_TERMINAL_AUDIT
    origin=dict(role='native_qwen_preupdate_qualified_bank_origin_v1',base=old['base'],source_commit=old['source_commit'],
        source_map_sha256=helper.QUALIFIED_BANK_MAP,run_id=helper.QUALIFIED_BANK_RUN,execution_id=helper.QUALIFIED_BANK_EXECUTION,
        bank_base=old['bank_base'],complete_sha256=complete['sha256'],feature_manifest_sha256=item['metadata']['feature_manifest_sha256'],
        admitted_sha256=item['sha256'],admitted=item['metadata'],producer_source_sha256={name:old['source_sha256'][name] for name in helper.QUALIFIED_BANK_PRODUCERS},
        old_handle_sha256=base.digest(args.qualified_bank_origin_handle),old_intent_canonical_sha256=hashlib.sha256(base.canonical(old)).hexdigest(),
        bank_admitted_observation_sha256=base.digest(args.bank_admitted_receipt),terminal_audit=str(audit_path),
        terminal_audit_sha256=base.digest(audit_path),terminal_audit_bytes=audit_path.stat().st_size,
        formal_completed_outers=0,actor_or_head_updates_inherited=False)
    helper.validate_qualified_bank_origin(origin,mapping);helper.validate_terminal_bank_origin(origin,base.document(audit_path))
    return origin


def create(warmup,scaffold,base,args):
    human=base.document(warmup.HUMAN);warmup.validate_human(human);assert human['formal200_explicitly_authorized_by_latest_goal'] is True
    repo=args.source_repo.resolve();commit=args.source_commit
    assert re.fullmatch('[0-9a-f]{40}',commit) and subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==commit
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain']).strip()
    mapping=base.runtime_module().published_source(repo,commit);publication=base.document(args.source_publication)
    assert all(base.digest(repo/name)==value for name,value in mapping.items())
    assert all(name in mapping for name in ('reproduction/a800_common/qprobe_qwen_formal_ops.py','reproduction/a800_common/qprobe_token_feature_bank.py','reproduction/common8/qprobe_qwen_bind.py'))
    assert publication['source_commit']==publication['remote_commit']==commit and publication['remote_fresh_object_store_disk_git_hash_equal'] is True
    assert publication['source_map_sha256']==hashlib.sha256(base.canonical(mapping)).hexdigest()
    origin=qualified_bank_origin(base,args,repo,mapping)
    bundle=args.bundle.resolve();subprocess.run(['git','-C',str(repo),'bundle','verify',str(bundle)],capture_output=True,check=True)
    assert subprocess.check_output(['git','-C',str(repo),'bundle','list-heads',str(bundle)],text=True).split()[0]==commit
    qualified=base.document(args.warmup_handle)
    assert qualified['role']==warmup.ROLE and qualified['native_block_class']==warmup.NATIVE_CLASS and qualified['warmup_epochs']==10 and qualified['expected_head_adam_updates']==200
    assert re.fullmatch('/data/shaoshitong/Search-G1-qprobe-qwen-lastblock-warmup10-[0-9a-f]{32}',qualified['base'])
    assert (qualified['base'],qualified['source_commit'],hashlib.sha256(base.canonical(qualified['source_sha256'])).hexdigest())==(QUALIFIED_WARMUP_BASE,QUALIFIED_WARMUP_SOURCE,QUALIFIED_WARMUP_MAP)
    identifier=uuid.uuid4().hex;bank_identifier=uuid.uuid4().hex;old=qualified['old_intent']
    intent=dict(role=ROLE,host='baai-a800',deployment_id=identifier,base=PREFIX+identifier,created_unix=time.time(),source_commit=commit,source_sha256=mapping,published_ref=publication['published_ref'],source_repo=str(repo),
        source_publication=str(args.source_publication.resolve()),source_publication_bytes=args.source_publication.stat().st_size,source_publication_sha256=base.digest(args.source_publication),bundle=str(bundle),bundle_bytes=bundle.stat().st_size,bundle_sha256=base.digest(bundle),
        human_intent_sha256=base.digest(warmup.HUMAN),native_formal_200_authorized=True,source_publication_authorized=True,target_updates=200,head_device='cuda:0',
        warmup_handle=qualified,warmup_handle_sha256=base.digest(args.warmup_handle),warmup_base=qualified['base'],warmup_source_commit=qualified['source_commit'],warmup_source_map_sha256=hashlib.sha256(base.canonical(qualified['source_sha256'])).hexdigest(),
        bank_base=origin['bank_base'] if origin is not None else '/data/shaoshitong/Search-G1-qprobe-fulltoken-feature-bank-'+bank_identifier,baseline_base=old['baseline_base'],baseline_commit=old['baseline_commit'],baseline_intent_sha256=old['baseline_intent_sha256'],initial_probes_source_base='/data/shaoshitong/Search-G1-a800-fresh200-7895330a2c584bba9940720e98f006ef',
        parent_intent=qualified['parent_intent'],headfit_resume_only=True,resume_token_feature_output=qualified['resume_token_feature_output'],cleanup_library_repo=qualified['cleanup_library_repo'],cleanup_library_source=qualified['cleanup_library_source'],cleanup_library_source_sha256=qualified['cleanup_library_source_sha256'],no_automatic_retry=True)
    if origin is not None:intent['qualified_bank_origin']=origin
    authority(warmup,base,intent);handle=args.handle.resolve() if args.handle else ROOT/'preparation'/('qprobe-qwen-native-formal-'+identifier+'.json')
    base.once(handle,intent);print(json.dumps(dict(handle=str(handle),base=intent['base'],bank_base=intent['bank_base'],source_commit=commit,warmup_base=intent['warmup_base'])))


def perform(warmup,scaffold,base,args):
    commit=subprocess.check_output(['git','-C',str(OPS),'rev-parse','HEAD'],text=True).strip();relative=str(Path(__file__).resolve().relative_to(OPS))
    assert subprocess.check_output(['git','-C',str(OPS),'show',commit+':'+relative])==Path(__file__).read_bytes(), 'Publish exact formal operator first'
    handle=args.handle.resolve();intent=base.document(handle);authority(warmup,base,intent)
    assert not (intent.get('qualified_bank_origin') is not None and args.action in ('bank-prepare','bank-encode')), 'Existing qualified bank must not be prepared or encoded again'
    payload=dict(intent=intent,operator_commit=commit,action=args.action,worker=base.WORKER,admission=args.admission)
    if args.action=='warmup-admit':assert args.warmup_receipt;payload['warmup_receipt']=base.document(args.warmup_receipt)
    if args.action=='launch':assert args.review;payload['review']=base.document(args.review)
    if args.action=='bank-encode':assert args.physical_index is not None;payload['physical_index']=args.physical_index
    code=stage_code(warmup,scaffold,base) if args.action=='stage' else remote_code(base);ast.parse(code)
    if args.action!='observe':base.once(handle.with_name(handle.stem+'.'+args.action+'.claim.json'),dict(action=args.action,source_commit=intent['source_commit'],operator_commit=commit,claimed_unix=time.time(),no_automatic_retry=True))
    if args.action=='stage':
        stream=io.BytesIO()
        with tarfile.open(fileobj=stream,mode='w') as archive:
            raw=base.canonical(payload);item=tarfile.TarInfo('request.json');item.size=len(raw);archive.addfile(item,io.BytesIO(raw))
            inputs=[(Path(intent['bundle']),'preparation/published-source.bundle',intent['bundle_sha256']),(Path(intent['source_publication']),'preparation/source-publication.json',intent['source_publication_sha256'])]
            if intent.get('qualified_bank_origin') is not None:
                origin=intent['qualified_bank_origin'];inputs.append((Path(origin['terminal_audit']),'preparation/qualified-bank-terminal-audit.json',origin['terminal_audit_sha256']))
            for path,name,expected in inputs:
                assert base.digest(path)==expected
                with path.open('rb') as source:item=tarfile.TarInfo(name);item.size=path.stat().st_size;archive.addfile(item,source)
        data=stream.getvalue()
    else:data=base.canonical(payload)
    route=base.runtime_module().ssh_command(code);assert route[-1].startswith('python3 -c ');route[-1]=base.PYTHON+' -c '+route[-1][len('python3 -c '):]
    receipt=args.receipt.resolve() if args.receipt else handle.with_name(handle.stem+'.'+args.action+'-'+str(time.time_ns())+'.json');result=dict(action=args.action,operator_commit=commit,source_commit=intent['source_commit'],base=intent['base'],no_automatic_retry=True,observation=None)
    with tempfile.TemporaryFile() as out,tempfile.TemporaryFile() as err:
        try:
            done=subprocess.run(route,input=data,stdout=out,stderr=err,timeout=400 if args.action=='stage' else 120);result['exit_code']=done.returncode
        except (OSError,subprocess.TimeoutExpired) as failure:result.update(exit_code=None,failure_type=type(failure).__name__,status='TRANSPORT_UNKNOWN_OBSERVE_SAME_HANDLE')
        result.update(stdout_bytes=out.tell(),stderr_bytes=err.tell());out.seek(0);raw=out.read(1024*1024+1);err.seek(0);errors=err.read(256*1024)
        result['bounded_stderr_sha256']=hashlib.sha256(errors).hexdigest();result['remote_exception_types']=sorted(set(value.decode() for value in re.findall(rb'(?m)^([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception))(?::|$)',errors)))
        result['remote_frames']=[dict(file=Path(name.decode(errors='replace')).name,line=int(line)) for name,line in re.findall(rb'File "([^"]+)", line ([0-9]+)',errors)[-8:]]
    if len(raw)<=1024*1024:
        rows=[json.loads(line.split(':',1)[1]) for line in raw.decode(errors='strict').splitlines() if line.startswith('TOKEN_WARMUP_JSON:')];assert len(rows)<=1
        if rows:result['observation']=rows[0]
    result['checked_unix']=time.time();base.once(receipt,result);print(json.dumps(dict(receipt=str(receipt),**result),allow_nan=False))
    return 0 if result['exit_code']==0 and result['observation'] is not None else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('create','stage','prepare','warmup-admit','bank-prepare','bank-encode','bank-admit','bind','adopt-initial0','prepare-launch','launch','observe'))
    parser.add_argument('--handle',type=Path);parser.add_argument('--receipt',type=Path);parser.add_argument('--source-repo',type=Path,default=RUNTIME);parser.add_argument('--source-commit');parser.add_argument('--source-publication',type=Path);parser.add_argument('--bundle',type=Path)
    parser.add_argument('--warmup-handle',type=Path);parser.add_argument('--warmup-receipt',type=Path);parser.add_argument('--physical-index',type=int);parser.add_argument('--review',type=Path);parser.add_argument('--admission',action='store_true')
    parser.add_argument('--qualified-bank-origin-handle',type=Path);parser.add_argument('--bank-admitted-receipt',type=Path);parser.add_argument('--terminal-audit-receipt',type=Path)
    args=parser.parse_args();assert not args.admission or args.action=='observe';warmup,scaffold,base=support()
    if args.action=='create':
        assert all(value is not None for value in (args.source_commit,args.source_publication,args.bundle,args.warmup_handle));return create(warmup,scaffold,base,args)
    assert args.handle is not None;return perform(warmup,scaffold,base,args)


if __name__=='__main__':raise SystemExit(main())
