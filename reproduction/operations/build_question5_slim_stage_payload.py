"""Local-only payload builder for the published138cf9 question5 slim pipeline.

Reads code through read-only Git archive/show metadata; never calls a remote,
stages a job, writes Git, changes an old handle or creates an automation.
Only source/config/small manifest metadata are persisted on this computer.
"""
import argparse
import base64
import copy
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile
import time
import uuid

ROOT=Path('/Users/shaoshitong/Documents/Search-G1')
REPO=ROOT/'repo'
PREP=ROOT/'preparation'
COMMIT='138cf9bc68488a4f778d8f59c4293cc208f18731'
TREE='cdf486b049188e396c254e667d61d31678209a31'
PREFIX='/data/shaoshitong/Search-G1-question5-ema-joint200-20261016-'
A800='/data/shaoshitong/Search-G1-a800-fresh200-0850db2977bd4801b9ced300b0ab387f'
ORIGINAL_ARCHIVE_SHA='085b039051359f341bcc123a952e2ac9500fd890331dc4f7cd2e367c81d39ced'
CANONICAL_SHA='5d40180ba54d35513bb780408caa068d7d60123214a49b0739f835a2ae47c8a1'


def digest(raw):return hashlib.sha256(raw).hexdigest()
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()+b'\n'
def load(path):return json.loads(Path(path).read_bytes())
def sha(path):return digest(Path(path).read_bytes())


def exclusive(path,raw):
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())


def code_archive(publication):
    if (publication['source_commit']!=COMMIT or publication['source_tree']!=TREE or
            publication.get('all_remote_source_blobs_matched') is not True):
        raise ValueError('Use the actual published138cf9 source proof')
    tree=subprocess.check_output(['git','-C',str(REPO),'rev-parse',COMMIT+'^{tree}'],text=True).strip()
    if tree!=TREE:raise ValueError('Pinned Git tree identity changed')
    archive=subprocess.check_output(['git','-C',str(REPO),'archive','--format=tar.gz',COMMIT])
    if len(archive)>32*1024**2:raise ValueError('Source archive exceeds bounded local source allowance')
    files={}
    with tarfile.open(fileobj=io.BytesIO(archive),mode='r:gz') as source:
        for item in source:
            path=PurePosixPath(item.name)
            if path.is_absolute() or '..' in path.parts or item.issym() or item.islnk():raise ValueError('Unsafe source archive member')
            if item.isdir():continue
            if not item.isfile() or item.name in files or item.size>64*1024**2:raise ValueError('Invalid source archive member')
            raw=source.extractfile(item).read();files[item.name]=digest(raw)
    for name,expected in publication['files'].items():
        if files.get(name)!=expected:raise ValueError('Published source blob differs: '+name)
    operator='reproduction/operations/a800_question5_slim_warmup_control.py'
    if files.get(operator)!=publication['files'][operator]:raise ValueError('Missing published slim controller')
    return archive,files


def mapped_transfer(original,base,h100):
    expected=dict(role='PREPARED_REFERENCE_PREFIX5_ARCHIVE',prefix_tokens=5,
        encoder_revision='aa8e72537993ba99e69dfaafa59ed015b17504d1',train_groups=348160,
        train_trajectories=1740800,test_groups=512,test_trajectories=2560,sources=68,
        canonical_train_sha256=CANONICAL_SHA,canonical_questions=169028,source_archive_sha256=ORIGINAL_ARCHIVE_SHA)
    if any(original.get(key)!=value for key,value in expected.items()):raise ValueError('Wrong complete H100 transfer manifest identity')
    for key in ('files','reference_manifest_sha256','internal_manifest_sha256','concatenated_sha256','parts','bytes','compressed_bytes','question_max_tokens'):
        if key not in original:raise ValueError('Full transfer manifest is required; missing '+key)
    for key in ('reference_manifest_sha256','internal_manifest_sha256','concatenated_sha256'):
        if re.fullmatch('[0-9a-f]{64}',original[key]) is None:raise ValueError('Invalid transfer SHA field')
    if not original['files'] or not original['parts']:raise ValueError('Empty complete H100 package')
    result=copy.deepcopy(original);names=set()
    for before,after in zip(original['parts'],result['parts']):
        source=PurePosixPath(before['path']);name=source.name
        if (not source.is_absolute() or not str(source).startswith(h100['base']+'/derived/') or
                '..' in source.parts or name in names or re.fullmatch(r'[A-Za-z0-9_.-]+',name) is None or
                type(before['bytes']) is not int or before['bytes']<=0 or re.fullmatch('[0-9a-f]{64}',before['sha256']) is None):
            raise ValueError('Unsafe, duplicate or unbound H100 transfer part')
        names.add(name);after['path']=base+'/source-archives/'+name
    if sum(item['bytes'] for item in original['parts'])!=original['bytes'] or original['bytes']!=original['compressed_bytes']:
        raise ValueError('Part bytes do not cover the full derived archive')
    check=copy.deepcopy(result)
    for before,after in zip(original['parts'],check['parts']):after['path']=before['path']
    if check!=original:raise AssertionError('Only parts[].path may change')
    return result


def describe():
    return dict(role='local_only_question5_slim_payload_builder',source_commit=COMMIT,source_tree=TREE,
        input='full H100 derived/transfer-manifest.json; no part/concat-only shortcut',
        stage_payload_keys=['source_archive_b64','side_files'],outputs=['new handle JSON','stage payload JSON','original transfer assets JSON','small binding JSON'],
        prepared_reference_archive='full original transfer object with only parts[].path remapped to new BASE/source-archives/basename',
        source_receipt='generated by published STAGE; not a caller-supplied acceptance',
        execution_contract='published worker/fit_config; controller has no additional required execution_contract payload key',
        no_stage_or_launch=True,no_remote_or_git_mutation=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--transfer-manifest',type=Path)
    parser.add_argument('--output-prefix',type=Path)
    parser.add_argument('--publication',type=Path,default=PREP/'question5-slim-source-publication-20261016.json')
    parser.add_argument('--old-handle',type=Path,default=PREP/'question5-ema-joint200-deployment-20261016.json')
    parser.add_argument('--human-intent',type=Path,default=PREP/'question5-ema-joint200-human-intent-20261016.json')
    parser.add_argument('--a800-assets',type=Path,default=PREP/'question5-remote-asset-binding-20261016.json')
    parser.add_argument('--a800-audit',type=Path,default=PREP/'qprobe-prefix-scale20-final-audit-1792141000881306000.json')
    parser.add_argument('--h100-handle',type=Path,default=PREP/'h100-question5-reference5-deployment-20261016.json')
    parser.add_argument('--describe',action='store_true');args=parser.parse_args()
    if args.describe:print(json.dumps(describe(),sort_keys=True));return
    if args.transfer_manifest is None:parser.error('--transfer-manifest is required')
    original_raw=args.transfer_manifest.read_bytes();original=json.loads(original_raw)
    old=load(args.old_handle);human_raw=args.human_intent.read_bytes();human=json.loads(human_raw)
    if (human['prefix_tokens']!=5 or human['epochs']!=200 or human['ema_decay']!=.99 or
            human['loss_weights']!=dict(trajectory_bce=1,mixed_group_bce=1) or human['new_rollout_authorized_or_required'] is not False):
        raise ValueError('Latest authorized scientific intent differs')
    fit=copy.deepcopy(old['fit_config'])
    if fit!=dict(schema_version=1,epochs=200,prefix_tokens=5,world_size=4,seed=1729,
            global_group_pattern=[51,51,51,51,52],micro_groups=6,ema_decay=.99,lr_start=3e-5,lr_end=3e-6,save_every_epochs=10):
        raise ValueError('Preserve the already published fit recipe exactly')
    audit_raw=args.a800_audit.read_bytes();audit=json.loads(audit_raw);proof=audit['audit'];assets=load(args.a800_assets)
    if (audit['accepted'] is not True or proof['accepted'] is not True or proof['run_id']!='62407ad0ed404586998439ca1f55b902' or
            proof['execution_id']!='dd3032e313e14ce2b9c8fea996aaf721' or proof['training_trajectories']!=512000 or
            proof['only_owned_cuda']!=[] or proof['producer_full_sha_proofs_rechecked'] is not True):
        raise ValueError('Require the actual accepted0850 data/feature audit')
    h100=load(args.h100_handle)
    if h100['source_commit']!=COMMIT:raise ValueError('Wrong H100 derivative source commit')
    run_id=uuid.uuid4().hex;execution_id=uuid.uuid4().hex;base=PREFIX+run_id
    mapped=mapped_transfer(original,base,h100)
    archive,files=code_archive(load(args.publication))
    plan=dict(schema_version=1,prefix_tokens=5,prepared_reference_archive=mapped,
        a800_dataset=dict(dataset_dir=A800+'/dataset',compact_manifest_sha256=proof['compact_manifest_sha256'],
            feature_composite_manifest=dict(path=A800+'/features/composite-manifest.json',sha256=proof['composite_manifest_sha256'])),
        reuse_cache=dict(path=assets['reuse']['cache_dir'],manifest_sha256=assets['reuse']['manifest_sha256']),
        cache_acceptance=dict(path=base+'/preparation/a800-accepted-final-audit.json',sha256=digest(audit_raw)))
    plan_raw=canonical(plan)
    handle=dict(schema_version=1,role='fresh_question5_joint200_slim_pipeline',base=base,run_id=run_id,execution_id=execution_id,
        prefix_tokens=5,epochs=200,source_commit=COMMIT,source_tree=TREE,source_archive_sha256=digest(archive),source_files=files,
        source_plan_sha256=digest(plan_raw),model_dir=old['model_dir'],human_intent_sha256=digest(human_raw),fit_config=fit,
        actor_training_budget=0,new_rollout_budget=0,created_unix=time.time(),
        original_h100_transfer_manifest_sha256=digest(original_raw),mapped_transfer_manifest_sha256=digest(canonical(mapped)),
        payload_builder_sha256=sha(__file__),previous_pretraining_handle_sha256=sha(args.old_handle))
    side={'source-plan.json':plan_raw,'a800-accepted-final-audit.json':audit_raw,'human-intent.json':human_raw,
          'h100-transfer-manifest-original.json':original_raw,'h100-transfer-manifest-a800.json':canonical(mapped)}
    payload=dict(source_archive_b64=base64.b64encode(archive).decode(),side_files={name:base64.b64encode(raw).decode() for name,raw in side.items()})
    prefix=args.output_prefix or PREP/('question5-slim-'+run_id)
    outputs={key:Path(str(prefix)+suffix) for key,suffix in (('handle','.handle.json'),('payload','.payload.json'),('assets','.transfer-assets.json'),('binding','.binding.json'))}
    if any(path.exists() for path in outputs.values()):raise FileExistsError('Keep prior local handles and payloads; choose a new output prefix')
    raw_handle=canonical(handle);raw_payload=canonical(payload)
    binding=dict(role='local_prepared_question5_slim_stage_payload_not_executed',base=base,run_id=run_id,execution_id=execution_id,
        source_commit=COMMIT,source_tree=TREE,source_files=len(files),source_archive_bytes=len(archive),source_archive_sha256=digest(archive),
        source_files_map_sha256=digest(canonical(files)),source_plan_sha256=digest(plan_raw),
        original_transfer_manifest_sha256=digest(original_raw),mapped_transfer_manifest_sha256=digest(canonical(mapped)),
        transfer_fields_preserved_except_part_paths=True,parts=len(original['parts']),derived_data_bytes=original['bytes'],
        a800_accepted_audit_bytes=len(audit_raw),a800_accepted_audit_sha256=digest(audit_raw),
        handle_sha256=digest(raw_handle),payload_sha256=digest(raw_payload),outputs={k:str(v) for k,v in outputs.items()},
        stage_executed=False,launch_executed=False,handoff_or_automation_written=False,created_unix=time.time())
    for key,raw in (('handle',raw_handle),('payload',raw_payload),('assets',original_raw),('binding',canonical(binding))):exclusive(outputs[key],raw)
    print(json.dumps(binding,sort_keys=True))


if __name__=='__main__':main()
