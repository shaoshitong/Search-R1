"""One compatible fresh-head continuation from the completed a799 token cache.

The exact published predecessor supplies transport, source verification, fixed
streaming staging, one-shot worker and bounded scalar observer. This entry has
no stop/cleanup mutation and forces the head-fit-only runner route. Qualify the
failed parent, CPU-prepare the new namespace, review ready evidence, then run.
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
OPS=Path(__file__).resolve().parents[2]
ROOT=Path('/Users/shaoshitong/Documents/Search-G1')
BASE_PATH='reproduction/operations/a800_qprobe_token_binary_warmup10.py'
BASE_COMMIT='cae8189c09992ff4be915dccf174eff9c6fbbf47'
BASE_SHA='ca079eea119cf160810b2e789ba8a3ab72fc7b5662efb5cac3ec38175f473e4e'
PARENT_HANDLE=ROOT/'preparation/qprobe-token-binary-warmup10-deployment-v2.json'
PARENT_HANDLE_SHA='c8073bb020900f9711833110a9211ea466696596adcf3fd47a73982c1ad0d620'
PARENT_BASE='/data/shaoshitong/Search-G1-qprobe-token-binary-warmup10-a799f7aff3e94045b64791655b878c31'
PARENT_SOURCE='bed0f379f6f65f0695fdc2e24b72818edf1c61c9'
NEW_RUNTIME=ROOT/'qprobe-token-binary-warmup10-headfit-resume'


def load_base():
    raw=(OPS/BASE_PATH).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==BASE_SHA
    assert subprocess.check_output(['git','-C',str(OPS),'show',BASE_COMMIT+':'+BASE_PATH])==raw
    spec=importlib.util.spec_from_file_location('frozen_token_binary_operations',OPS/BASE_PATH)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


ADDON = r'''
parent=Path('/data/shaoshitong/Search-G1-qprobe-token-binary-warmup10-a799f7aff3e94045b64791655b878c31')
assert intent['headfit_resume_only'] is True and intent['resume_token_feature_output']==str(parent/'warmup')
assert document(parent/'preparation/intent.json')==intent['parent_intent']
assert intent['parent_intent']['source_commit']=='bed0f379f6f65f0695fdc2e24b72818edf1c61c9'
source(parent/'repo',intent['parent_intent']['source_commit'],intent['parent_intent']['source_sha256'])
def audit():
 verify,_,_,_,_=isolated_library();saved=document(parent/'logs/old-stop-intent.json')
 assert saved['run_id']==old['run_id'] and saved['execution_id']==old['execution_id']
 actual=verify(oldbase,old,preserved_births=saved['preserved_births']);assert actual['accepted'] is True
 return actual
_original_runner=runner
def runner(gpu=None):
 return _original_runner(gpu)+['--resume-token-feature-output',str(parent/'warmup')]
def parent_qualification():
 verify,_,survivors,contexts,_=isolated_library()
 worker=dict(pid=145985,start_ticks=614484748,sid=145985,uid=os.getuid())
 producer=dict(pid=145986,start_ticks=614484752,sid=145985,uid=os.getuid())
 launch=document(parent/'logs/run-launch.json');claim=document(parent/'warmup/run-claim.json')
 assert launch['process']==worker and {key:claim[key] for key in producer}==producer
 assert launch['source_commit']==claim['task_config']['source_commit']==intent['parent_intent']['source_commit']
 assert launch['physical_index']==0 and launch['gpu_uuid']=='GPU-9b8498ae-93e5-dee7-545a-25669080cf83'
 assert not process(worker)['alive'] and not process(producer)['alive']
 live,reused_sessions=survivors([worker]);assert not live and not reused_sessions
 rows=list(csv.reader(io.StringIO(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True,timeout=20))))
 remaining,reused_pids=contexts([worker,producer],rows);assert not remaining and not reused_pids
 exit_record=document(parent/'logs/run-exit.json');failure=document(parent/'warmup/run-failure.json')
 assert exit_record['exit_code']==1 and sha(parent/'logs/run-exit.json')=='6af047d6fa270727d86d4e1ecbd96df5cfd22fa37ecab2b745c4e2502457bbd0'
 assert sha(parent/'warmup/run-failure.json')=='3f26f96ddab67a086e2a3c627a4018b9b2917130f096eecdf0e6224645f38900'
 assert failure['failure_type']=='AssertionError' and {key:failure[key] for key in producer}==producer
 assert failure['frames']==[dict(file='qprobe_token_warmup.py',function='main',line=304),dict(file='qprobe_token_warmup.py',function='run',line=246)]
 for relative in ('ready.json','warmup-metrics.json','fresh-head.pt','fresh-head.partial.pt'):
  assert not (parent/'warmup'/relative).exists() and not (parent/'warmup'/relative).is_symlink()
 progress=document(parent/'warmup/progress.json')
 assert progress['stage']=='token_feature_encoding' and progress['completed_questions']==progress['total_questions']==5120
 prepared=document(parent/'warmup/prepared.json');manifest_path=parent/'warmup/token-features/manifest.json';manifest=document(manifest_path)
 assert prepared['source_sha256']==intent['parent_intent']['source_sha256'] and prepared['task_config']==claim['task_config']
 assert manifest['count']==5120 and manifest['feature_dim']==2048 and manifest['representation']=='all_nonpadding_question_tokens_packed_v1'
 assert manifest['spec_sha']==hashlib.sha256(canonical(manifest['spec'])).hexdigest()
 assert manifest['spec']['global_max_tokens']==prepared['global_scan']['global_max_tokens']==244
 assert manifest['spec']['teacher_identity_sha256']==intent['teacher_identity_sha256']
 assert manifest['spec']['token_feature_producer_sha256']==intent['parent_intent']['source_sha256']['search_g1/qprobe_token_features.py']
 assert set(manifest['files'])=={'packed_features.npy','offsets.npy','lengths.npy','index.jsonl'}
 for name,item in manifest['files'].items():
  path=manifest_path.parent/name;assert path.resolve()==path and not path.is_symlink() and path.stat().st_uid==os.getuid() and path.stat().st_size==item['bytes']
 old_cleanup=audit()
 return dict(stage='FAILED_PARENT_TOKEN_CACHE_HEADFIT_CONTINUATION_QUALIFIED',metadata_only=True,no_remote_mutation=True,
  parent_base=str(parent),parent_source_commit=intent['parent_intent']['source_commit'],worker=worker,runner=producer,
  both_pinned_births_exited=True,private_session_survivors=0,owned_cuda_contexts_remaining=0,reused_sessions=[],reused_pids=[],
  observed_failure_type='AssertionError',observed_failure_line=246,head_adam_updates=0,
  completed_encoded_questions=5120,token_feature_manifest_sha256=sha(manifest_path),token_feature_spec_sha256=manifest['spec_sha'],
  token_feature_declared_files=manifest['files'],full_feature_sha_verification_delegated_to_cpu_prepare=True,
  old_actual_cleanup=old_cleanup,checked_unix=time.time())
'''


def program(base,action):
    if action=='stage':
        direct=r'''env=dict(os.environ,GIT_TERMINAL_PROMPT='0');direct=False
with (base/'preparation/github-clone.log').open('xb') as log:
 try:
  direct=subprocess.run(['git','clone','--no-checkout','--single-branch','--branch',intent['published_ref'].removeprefix('refs/heads/'),'https://github.com/shaoshitong/Search-R1.git',str(repo)],stdout=log,stderr=subprocess.STDOUT,env=env,timeout=90).returncode==0
 except subprocess.TimeoutExpired:pass
'''
        prior=r'''env=dict(os.environ,GIT_TERMINAL_PROMPT='0');direct=False
prior=Path('/data/shaoshitong/Search-G1-qprobe-token-binary-warmup10-a799f7aff3e94045b64791655b878c31')
assert intent['headfit_resume_only'] is True and intent['resume_token_feature_output']==str(prior/'warmup')
assert document(prior/'preparation/intent.json')==intent['parent_intent']
assert intent['parent_intent']['source_commit']=='bed0f379f6f65f0695fdc2e24b72818edf1c61c9'
prior_stage=document(prior/'logs/staged.json')
assert sha(prior/'logs/staged.json')=='4a6d4bd3a01d5f60975b04485dcbad2f6a304e3da452a0b2623c48dc6d50e4d4'
assert prior_stage['base']==str(prior) and prior_stage['source_commit']==intent['parent_intent']['source_commit']
assert prior_stage['source_transport']=='published_git_bundle_relay' and prior_stage['stage']=='PUBLISHED_SOURCE_STAGED_CPU_ONLY'
assert prior_stage['source_map_sha256']==hashlib.sha256(canonical(intent['parent_intent']['source_sha256'])).hexdigest()
once(base/'logs/prior-published-bundle-transport.json',dict(prior_base=str(prior),prior_stage_sha256=sha(prior/'logs/staged.json'),new_bundle_sha256=intent['bundle_sha256'],reason='actual_prior_direct_clone_failure_preserved',checked_unix=time.time()))
'''
        assert base.STAGE.count(direct)==1
        return base.STAGE.replace(direct,prior,1)
    common=base.COMMON+'\nbind(json.load(sys.stdin))\n'+base.OLD+'\n'+ADDON
    if action=='qualify-parent':return common+'\nemit(parent_qualification())\n'
    original=dict(prepare=base.PREPARE,observe=base.OBSERVE,**{'run-once':base.RUN})[action]
    assert original.startswith(base.COMMON)
    body=original[len(base.COMMON):]
    assert body.count('bind(json.load(sys.stdin));')==1
    body=body.replace('bind(json.load(sys.stdin));','',1)
    if action=='observe':
        assert body.count("'warmup/warmup-metrics.json'")==1
        body=body.replace("'warmup/warmup-metrics.json'","'warmup/partial-head.json','warmup/warmup-metrics.json'",1)
    if action=='run-once':
        assert body.count(base.OLD)==1;body=body.replace(base.OLD,'',1)
        # The head-only continuation must carry the exact reviewed preparation.
        body="ready=p['ready'];assert ready['role']=='token_binary_headfit_resume_reviewed_prepared_ready'\nassert ready['source_commit']==intent['source_commit'] and ready['human_intent_sha256']==intent['human_intent_sha256']\nassert ready['independent_code_review_approved'] is True and ready['resume_token_feature_output']==intent['resume_token_feature_output']\nassert ready['remote_prepared_sha256']==sha(output/'prepared.json') and ready['source_publication_sha256']==intent['source_publication_sha256']\n"+body
    if action!='observe':body='qualification=parent_qualification()\n'+body
    else:body+="\n"  # Original observer reads only existing bounded scalar/hash documents.
    return common+'\n'+body


def create(base,args):
    module=base.runtime_module();repo=args.source_repo.resolve();commit=args.source_commit
    assert re.fullmatch('[0-9a-f]{40}',commit)
    assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==commit
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain']).strip()
    mapping=module.published_source(repo,commit);publication=base.document(args.source_publication)
    assert all(base.digest(repo/name)==value for name,value in mapping.items())
    assert publication['source_commit']==publication['remote_commit']==commit and publication['remote_fresh_object_store_disk_git_hash_equal'] is True
    assert publication['source_map_sha256']==hashlib.sha256(base.canonical(mapping)).hexdigest()
    assert re.fullmatch('refs/heads/[A-Za-z0-9_./-]+',publication['published_ref'])
    bundle=args.bundle.resolve();subprocess.run(['git','-C',str(repo),'bundle','verify',str(bundle)],capture_output=True,check=True)
    assert subprocess.check_output(['git','-C',str(repo),'bundle','list-heads',str(bundle)],text=True).split()[0]==commit
    assert base.digest(PARENT_HANDLE)==PARENT_HANDLE_SHA
    parent=base.document(PARENT_HANDLE);assert parent['base']==PARENT_BASE and parent['source_commit']==PARENT_SOURCE
    identifier=uuid.uuid4().hex
    intent=dict(parent,base=base.PREFIX+identifier,deployment_id=identifier,created_unix=time.time(),
        source_commit=commit,source_sha256=mapping,published_ref=publication['published_ref'],source_repo=str(repo),
        source_publication=str(args.source_publication.resolve()),source_publication_bytes=args.source_publication.stat().st_size,
        source_publication_sha256=base.digest(args.source_publication),bundle=str(bundle),bundle_bytes=bundle.stat().st_size,bundle_sha256=base.digest(bundle),
        parent_intent=parent,parent_intent_sha256=PARENT_HANDLE_SHA,headfit_resume_only=True,
        resume_token_feature_output=PARENT_BASE+'/warmup',resume_scope='reuse_completed_token_features_then_fresh_head_10epochs_200Adam_only',
        frozen_base_operator_commit=BASE_COMMIT,frozen_base_operator_sha256=BASE_SHA)
    base.authority(intent)
    handle=args.handle.resolve() if args.handle else ROOT/'preparation'/('qprobe-token-binary-headfit-resume-'+identifier+'.json')
    base.once(handle,intent);print(json.dumps(dict(handle=str(handle),base=intent['base'],source_commit=commit,headfit_only=True)))


def perform(base,args):
    commit=subprocess.check_output(['git','-C',str(OPS),'rev-parse','HEAD'],text=True).strip()
    relative=str(Path(__file__).resolve().relative_to(OPS))
    assert subprocess.check_output(['git','-C',str(OPS),'show',commit+':'+relative])==Path(__file__).read_bytes(), 'Publish exact continuation operator first'
    handle=args.handle.resolve();intent=base.document(handle);base.authority(intent)
    assert intent['headfit_resume_only'] is True and intent['resume_token_feature_output']==PARENT_BASE+'/warmup'
    assert intent['parent_intent_sha256']==PARENT_HANDLE_SHA and base.document(PARENT_HANDLE)==intent['parent_intent']
    payload=dict(intent=intent,operator_commit=commit,worker=base.WORKER)
    if args.action=='run-once':
        assert args.physical_index is not None and args.ready is not None
        payload.update(physical_index=args.physical_index,ready=base.document(args.ready))
    code=program(base,args.action);ast.parse(code)
    if args.action not in ('observe','qualify-parent'):
        base.once(handle.with_name(handle.stem+'.'+args.action+'.claim.json'),dict(action=args.action,operator_commit=commit,source_commit=intent['source_commit'],claimed_unix=time.time(),no_automatic_retry=True,headfit_resume_only=True))
    if args.action=='stage':
        stream=io.BytesIO()
        with tarfile.open(fileobj=stream,mode='w') as archive:
            raw=base.canonical(payload);item=tarfile.TarInfo('request.json');item.size=len(raw);archive.addfile(item,io.BytesIO(raw))
            for path,name,expected in ((Path(intent['bundle']),'preparation/published-source.bundle',intent['bundle_sha256']),(Path(intent['source_publication']),'preparation/source-publication.json',intent['source_publication_sha256'])):
                assert base.digest(path)==expected
                with path.open('rb') as source:
                    item=tarfile.TarInfo(name);item.size=path.stat().st_size;archive.addfile(item,source)
        data=stream.getvalue()
    else:data=base.canonical(payload)
    route=base.runtime_module().ssh_command(code);assert route[-1].startswith('python3 -c ')
    route[-1]=base.PYTHON+' -c '+route[-1][len('python3 -c '):]
    receipt=args.receipt.resolve() if args.receipt else handle.with_name(handle.stem+'.'+args.action+'-'+str(time.time_ns())+'.json')
    result=dict(action=args.action,operator_commit=commit,source_commit=intent['source_commit'],base=intent['base'],headfit_resume_only=True,no_automatic_retry=True,observation=None)
    with tempfile.TemporaryFile() as out,tempfile.TemporaryFile() as err:
        try:
            done=subprocess.run(route,input=data,stdout=out,stderr=err,timeout=400 if args.action=='stage' else 120)
            result['exit_code']=done.returncode
        except (OSError,subprocess.TimeoutExpired) as failure:result.update(exit_code=None,failure_type=type(failure).__name__,status='TRANSPORT_UNKNOWN_OBSERVE_SAME_HANDLE')
        result.update(stdout_bytes=out.tell(),stderr_bytes=err.tell());out.seek(0);raw=out.read(512*1024+1);err.seek(0);errors=err.read(256*1024)
        result['bounded_stderr_sha256']=hashlib.sha256(errors).hexdigest()
        result['remote_exception_types']=sorted(set(value.decode() for value in re.findall(rb'(?m)^([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception))(?::|$)',errors)))
        result['remote_frames']=[dict(file=Path(name.decode(errors='replace')).name,line=int(line)) for name,line in re.findall(rb'File "([^"]+)", line ([0-9]+)',errors)[-8:]]
    if len(raw)<=512*1024:
        rows=[json.loads(row.split(':',1)[1]) for row in raw.decode(errors='strict').splitlines() if row.startswith('TOKEN_WARMUP_JSON:')]
        assert len(rows)<=1
        if rows:result['observation']=rows[0]
    result['checked_unix']=time.time();base.once(receipt,result)
    print(json.dumps(dict(receipt=str(receipt),**result),allow_nan=False))
    return 0 if result['exit_code']==0 and result['observation'] is not None else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('create','stage','qualify-parent','prepare','observe','run-once'))
    parser.add_argument('--handle',type=Path);parser.add_argument('--receipt',type=Path)
    parser.add_argument('--source-repo',type=Path,default=NEW_RUNTIME);parser.add_argument('--source-commit')
    parser.add_argument('--source-publication',type=Path);parser.add_argument('--bundle',type=Path)
    parser.add_argument('--ready',type=Path);parser.add_argument('--physical-index',type=int)
    args=parser.parse_args();base=load_base()
    if args.action=='create':
        assert all(value is not None for value in (args.source_commit,args.source_publication,args.bundle));return create(base,args)
    assert args.handle is not None
    return perform(base,args)


if __name__=='__main__':raise SystemExit(main())
