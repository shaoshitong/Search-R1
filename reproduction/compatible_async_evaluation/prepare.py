"""CPU-only remote preparation from real saved receipts; never submit or load CUDA."""
import argparse
import json
import os
import re
from urllib.parse import urlparse
from pathlib import Path
import shlex
import socket
import sys
import uuid
from reproduction.compatible_async_evaluation.contract import PROTOCOL_PATH,digest,protocol,verify_source_tree
from reproduction.compatible_async_evaluation.runtime import ROLE,validate_binding,checkpoint_hf_names
from reproduction.compatible_async_evaluation.journal import make_binding


def once(path,value):
    with Path(path).open('x') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())


def prepare(args, *, resume_plan=None):
    if not sys.platform.startswith('linux'):raise ValueError('COMPAT_PREPARATION_MUST_STAY_REMOTE_LINUX')
    base=Path(args.base).resolve()
    control=Path(__file__).resolve().parents[2];spec=protocol()
    if not base.is_absolute() or str(base).startswith('/Users/') or base==control:
        raise ValueError('COMPAT_SEPARATE_REMOTE_NAMESPACE_REQUIRED')
    publication=json.loads(Path(args.control_publication_proof).read_text())
    if publication.get('published_control_commit')!=args.control_commit or publication.get('all_blobs_full_sha_match') is not True:
        raise ValueError('COMPAT_PUBLISHED_CONTROL_FULL_SHA_PROOF_REQUIRED')
    mapping=publication.get('files',{})
    if not mapping or not any(n.startswith('reproduction/compatible_async_evaluation/') for n in mapping):
        raise ValueError('COMPAT_PUBLISHED_PACKAGE_SOURCE_MISSING')
    verify_source_tree(control,mapping)
    checkpoint=Path(args.checkpoint).resolve();receipt_path=Path(args.checkpoint_receipt).resolve()
    receipt=json.loads(receipt_path.read_text());step=args.step
    if step==0:
        if not args.run_id:raise ValueError('COMPAT_OFFICIAL_ZERO_LOGICAL_RUN_ID_REQUIRED')
        run_id=args.run_id;certified={r['file']:r['sha256'] for r in receipt['files']}
    else:
        if receipt.get('step')!=step or Path(receipt.get('checkpoint','')).resolve()!=checkpoint or receipt.get('status')!='QPROBE_COMPLETED_PAIR_FULL_SHA_CPU_STATES_PASS':
            raise ValueError('COMPAT_REAL_COMPLETE_CHECKPOINT_RECEIPT_REQUIRED')
        run_id=receipt['run_id'];certified={r['relative_path']:r['full_read_sha256'] for r in receipt['files'] if '/' not in r['relative_path']}
        if args.run_id and args.run_id!=run_id:raise ValueError('COMPAT_WRONG_EXPERIMENT_RUN_ID')
    hf=checkpoint_hf_names(checkpoint)
    files={n:certified[n] for n in hf}
    source_base=Path(args.source_base).resolve()
    from reproduction.common8.validation_identity import expected
    record,frame,ids=expected(source_base,args.split);sources=frame['data_source'].tolist()
    journal=make_binding(run_id,digest(receipt_path),record['sha256'],ids,sources,step,args.split)
    sharing=json.loads(Path(args.sharing_bindings).read_text())
    allowed_sharing={'allocation_mode','preserved_foreign_allocations','preserved_foreign_gpu_processes','forbidden_training_job_ids','gpu_lock_filename'}
    if not isinstance(sharing,dict) or not set(sharing)<=allowed_sharing:
        raise ValueError('COMPAT_SHARING_CANNOT_OVERRIDE_PROTECTED_BINDING_FIELDS')
    if sharing.get('gpu_lock_filename','{uuid}.lock')!='{uuid}.lock':
        raise ValueError('COMPAT_GLOBAL_UUID_LOCK_NAME_REQUIRED')
    if sharing.get('allocation_mode') not in ('exclusive','shared'):
        raise ValueError('COMPAT_EXPLICIT_SHARING_BINDING_REQUIRED')
    if sharing['allocation_mode']=='shared' and not isinstance(sharing.get('preserved_foreign_allocations'),list):
        raise ValueError('COMPAT_SHARED_FOREIGN_ALLOCATION_ALLOWLIST_REQUIRED')
    old_config=Path(args.reference_config).resolve()
    binding=dict(role=ROLE,run_id=run_id,checkpoint_step=step,split=args.split,checkpoint=str(checkpoint),
        checkpoint_manifest=str(receipt_path),checkpoint_manifest_sha256=digest(receipt_path),
        checkpoint_files_sha256=files,generation_config_sha256=files['generation_config.json'],
        official_model=str(Path(args.official_model).resolve()),resolved_config_sha256=digest(old_config),
        validation_sha256=record['sha256'],journal_binding=journal,
        recipe_id=spec['recipe_id'],recipe_sha256=digest(PROTOCOL_PATH),source_root=str(control),
        source_commit=args.control_commit,source_sha256=mapping,
        control_source_commit=args.control_commit,control_source_sha256=mapping,
        control_publication_proof_sha256=digest(args.control_publication_proof),retrieval_url=args.retrieval_url)
    # No large-file hashing or GPU load here. The isolated runner verifies HF
    # bytes before loading, and the future evaluator pipeline verifies assets.
    validate_binding(binding,old_config,verify_weights=False)
    env=Path(args.environment_root).resolve()
    if not (env/'bin/python').is_file():raise ValueError('COMPAT_EXISTING_REMOTE_ENVIRONMENT_REQUIRED')
    if len(args.gpu_uuid)!=8 or len(set(args.gpu_uuid))!=8:raise ValueError('COMPAT_ALL_EIGHT_EXPLICIT_UUIDS_REQUIRED')
    if not re.fullmatch('[0-9a-f]{32}',args.execution_id) or not re.fullmatch('[0-9a-f]{32}',run_id):raise ValueError('COMPAT_EXECUTION_OR_RUN_ID_INVALID')
    parsed=urlparse(args.retrieval_url)
    if parsed.scheme!='http' or parsed.hostname not in ('localhost','127.0.0.1') or parsed.path!='/retrieve' or not parsed.port or not 1024<=parsed.port<=65535:raise ValueError('COMPAT_LOCAL_RETRIEVER_URL_REQUIRED')
    if not Path(args.gpu_lock_directory).resolve().is_dir():raise ValueError('COMPAT_EXISTING_GLOBAL_UUID_LOCK_DIRECTORY_REQUIRED')
    reference=Path(spec['protocol_authoritative_source']['base'])/'logs/retrieval-admission.json'
    reference_sha=digest(reference);reference_value=json.loads(reference.read_text())
    if reference_value.get('accepted') is not True or reference_value.get('run_id')!=spec['protocol_authoritative_source']['historical_run'] or any(reference_value.get(k)!=spec['retrieval'][k] for k in ('index_sha256','corpus_sha256')):
        raise ValueError('COMPAT_OLD_RETRIEVAL_REFERENCE_INVALID')
    if any(not Path(path).is_file() for path in (args.retrieval_index,args.retrieval_corpus)) or not Path(args.retriever_gpu_lib).is_dir():raise ValueError('COMPAT_RETRIEVAL_REMOTE_ASSET_PATH_MISSING')
    for name in reference_value['encoder_assets']['files']:
        if Path(name).name!=name or not (Path(args.retrieval_encoder)/name).is_file():raise ValueError('COMPAT_E5_REMOTE_ASSET_PATH_MISSING')
    if base.exists() and any(base.iterdir()):raise ValueError('COMPAT_EXISTING_NAMESPACE_INSPECT_DO_NOT_REPLAY')
    base.mkdir(parents=True,exist_ok=True)
    for folder in ('preparation','logs','journal','tmp','cache'):(base/folder).mkdir()
    once(base/'preparation/prepare-intent.json',dict(role=ROLE,run_id=run_id,checkpoint_step=step,split=args.split,recipe_sha256=digest(PROTOCOL_PATH),execution_id=args.execution_id))
    if resume_plan is not None:
        once(base/'preparation/resume-intent.json',resume_plan)
    config=base/'preparation/reference-config.json';config.write_bytes(old_config.read_bytes())
    target=base/'preparation/binding.json';once(target,binding)
    (base/'env').symlink_to(env,target_is_directory=True)
    manifest=dict(role=ROLE,base=str(base),run_id=run_id,execution_id=args.execution_id,
        checkpoint_step=step,split=args.split,source_base=str(source_base),binding=str(target),config=str(config),
        journal=str(base/'journal'),control_repo=str(control),source_root=str(control),
        recipe_sha256=digest(PROTOCOL_PATH),world_size=8,cpu_request=96,memory_request='512G',
        physical_gpu_uuids=args.gpu_uuid,gpu_lock_directory=str(Path(args.gpu_lock_directory).resolve()),
        retrieval_url=args.retrieval_url,retrieval_index_path=str(Path(args.retrieval_index).resolve()),
        retrieval_corpus_path=str(Path(args.retrieval_corpus).resolve()),retrieval_encoder=str(Path(args.retrieval_encoder).resolve()),
        retriever_gpu_lib=str(Path(args.retriever_gpu_lib).resolve()),
        retrieval_server_source=str(control/'search_r1/search/retrieval_server.py'),
        retrieval_reference_admission=str(reference),retrieval_reference_admission_sha256=reference_sha,
        ray_tmp_dir='/tmp/g1ca-'+args.execution_id[:12],**sharing)
    if resume_plan is not None:
        manifest['resume_intent_sha256']=digest(base/'preparation/resume-intent.json')
    once(base/'preparation/compatible-evaluation-manifest.json',manifest)
    launch=base/'preparation/launch.sh'
    launch.write_text('#!/bin/bash\nset -euo pipefail\nexport PYTHONPATH='+shlex.quote(str(control))+'\nexec '+shlex.join([str(base/'env/bin/python'),'-m','reproduction.compatible_async_evaluation.pipeline','--base',str(base)])+'\n')
    launch.chmod(0o700)
    once(base/'preparation/prepare-receipt.json',dict(role=ROLE,run_id=run_id,checkpoint_step=step,split=args.split,
        binding_sha256=digest(target),manifest_sha256=digest(base/'preparation/compatible-evaluation-manifest.json'),
        actual_gpu_execution=False,submitted=False,canonical_evaluation_complete=False))
    return dict(base=str(base),launch=str(launch),submitted=False,actual_gpu_execution=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('base','control-publication-proof','checkpoint','checkpoint-receipt','source-base','official-model','reference-config','sharing-bindings','environment-root','gpu-lock-directory','retrieval-index','retrieval-corpus','retrieval-encoder','retriever-gpu-lib'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--control-commit',required=True);p.add_argument('--step',type=int,required=True,choices=(0,5,50,100,150,200))
    p.add_argument('--split',required=True,choices=('validation','final'));p.add_argument('--run-id')
    p.add_argument('--execution-id',default=uuid.uuid4().hex)
    p.add_argument('--gpu-uuid',action='append',required=True)
    p.add_argument('--retrieval-url',required=True)
    a=p.parse_args();print(json.dumps(prepare(a)))

if __name__=='__main__':main()
