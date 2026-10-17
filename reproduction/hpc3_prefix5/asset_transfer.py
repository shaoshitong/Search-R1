"""Remote-only, private-HF transfer of the exact reference Prefix5 warmup.

Read credentials from stdin once. Never persist them, signed URLs, or headers.
The source exports only the retained EMA and its original qualification metadata;
the original 1.24GB warmup checkpoint is preserved. No rollout tensors or cache
are needed to initialize the online probe.
"""
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import time

SOURCE_BASE = '/data/shaoshitong/search-g1-prefix-selection200-a800-prefix5-54ac33dcebe64e62a11fe4177f41f116'
SOURCE_SHA = 'b5c81ae76c12f677a0663d6e6b0f0482ca63fde0507c2a018aa6034282105594'
SOURCE_COMMIT = '79c645a4a51bd0bcb0fab4ee36b8fcd634290b2f'
UPLOAD_ROOT = '/data/shaoshitong/thread5-prefix5-hpc3-sync-20261017'
TARGET_ROOT = '/data/user/jxu957/experiments/dynrpo-prefix5-thread5-20261017'
REPOSITORY = 'Sutoonq/TMP_MODEL'
HF_PATH = 'dynrpo-prefix5-thread5-20261017/reference-assets.tar'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8*1024**2), b''):
            h.update(part)
    return h.hexdigest()


def publish(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp-'+str(os.getpid()))
    with temporary.open('x') as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def export(root):
    import torch
    source = Path(SOURCE_BASE)
    sys.path.insert(0, str(source/'repo'))
    from reproduction.prefix_selection200.prefix5_model import warmup_model_weights
    c = json.loads((source/'preparation/launch-config.json').read_text())
    checkpoint = Path(c['probe_source_checkpoint'])
    if c['source_commit'] != SOURCE_COMMIT or c['probe_source_sha256'] != SOURCE_SHA or sha(checkpoint) != SOURCE_SHA:
        raise ValueError('The retained reference warmup identity differs')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model, buffers, qualification = warmup_model_weights(saved, epoch=5, model_kind='ema')
    # Keep every scientific qualification field, excluding unused optimizer,
    # raw head and per-rank RNG tensors. The exact production extractor validates
    # this compact representation again before publication and target import.
    compact = {k:v for k,v in saved.items() if k in (
        'role','binding','binding_sha256','epoch','completed_epoch','next_update_in_epoch',
        'counters','actual_adam_steps_per_rank','actor_updates','epoch_metrics','epoch_metrics_sha256')}
    compact['ema'] = dict(saved['ema'], model=model, nonpersistent_buffers=buffers)
    compact['export_provenance'] = dict(source=str(checkpoint), source_sha256=SOURCE_SHA,
        source_commit=SOURCE_COMMIT, model_only=True, optimizer_imported=False)
    warmup_model_weights(compact, epoch=5, model_kind='ema')
    head = root/'warmup-epoch5-ema.pt'
    torch.save(compact, head)
    selected = {'warmup-epoch5-ema.pt':head,
        'reference-launch-config.json':source/'preparation/launch-config.json',
        'source-pool.json':Path(c['source_question_pool']),
        'fixed512-identities.json':Path(c['fixed_test_identity_path']),
        'composite-manifest.json':Path(c['composite_manifest_path']),
        'feature-manifest.json':Path(c['fixed_test_features'])/'manifest.json',
        'grounding/version-manifest.json':Path(c['grounding_manifest']),
        'calibration-300.parquet':Path(c['grounding_calibration_questions'])}
    grounding = json.loads(Path(c['grounding_manifest']).read_text())
    for key in ('boundary','commit'):
        selected['grounding/'+key+'_probe.pkl'] = Path(grounding[key])
    files = []
    for name, path in selected.items():
        if not path.is_file() or path.is_symlink():
            raise ValueError('Reference export requires regular immutable inputs')
        files.append(dict(path=name, source=str(path), bytes=path.stat().st_size, sha256=sha(path)))
    manifest = dict(role='REFERENCE_PREFIX5_EMA5_MODEL_ONLY_REMOTE_EXPORT',
        original_warmup_sha256=SOURCE_SHA, original_source_commit=SOURCE_COMMIT,
        warmup_qualification=qualification, files=files, created_unix=time.time())
    publish(root/'manifest.json', manifest)
    target = root/'reference-assets.tar'
    with tarfile.open(target, 'w') as archive:
        archive.add(root/'manifest.json', arcname='manifest.json', recursive=False)
        for row in files:
            archive.add(selected[row['path']], arcname=row['path'], recursive=False)
    return dict(manifest=manifest, archive=str(target), archive_bytes=target.stat().st_size,
        archive_sha256=sha(target), compact_warmup_sha256=sha(head))


def upload(token, root):
    from huggingface_hub import HfApi, CommitOperationAdd
    if str(root) != UPLOAD_ROOT:
        raise ValueError('Unexpected source export namespace')
    complete = root/'upload-complete.json'
    if complete.exists():
        return json.loads(complete.read_text())
    api = HfApi(token=token)
    if api.repo_info(REPOSITORY).private is not True:
        raise ValueError('The authorized staging repository must be private')
    proof_path = root/'export-complete.json'
    if proof_path.exists():
        proof=json.loads(proof_path.read_text())
        if sha(proof['archive']) != proof['archive_sha256']:
            raise ValueError('Retained export archive changed')
    else:
        publish(root/'progress.json', dict(stage='EXPORT_AND_SOURCE_SHA', epoch=time.time()))
        proof=export(root);publish(proof_path,proof)
    publish(root/'progress.json', dict(stage='HF_COMMIT_IN_PROGRESS', epoch=time.time(),
        archive_bytes=proof['archive_bytes'], compact_warmup_sha256=proof['compact_warmup_sha256']))
    commit=api.create_commit(REPOSITORY, commit_message='Private Prefix5 EMA5 model-only transfer to HPC3',
        operations=[CommitOperationAdd(path_in_repo=HF_PATH,path_or_fileobj=proof['archive'])])
    item=api.get_paths_info(REPOSITORY,[HF_PATH],revision=commit.oid)[0]
    lfs=item.lfs
    if item.size != proof['archive_bytes'] or lfs.sha256 != proof['archive_sha256']:
        raise ValueError('Official HF LFS size/SHA differs from the actual source archive')
    result=dict(proof,repo=REPOSITORY,path=HF_PATH,revision=commit.oid,
        status='SOURCE_AND_PRIVATE_HF_SHA_VERIFIED',completed_unix=time.time(),mac_model_data=False)
    publish(complete,result)
    return result


def download(token, root, proof):
    import requests
    if str(root) != TARGET_ROOT:
        raise ValueError('Unexpected target namespace')
    assets=root/'reference-assets';assets.mkdir(exist_ok=True)
    complete=root/'transfer/download-complete.json'
    if complete.exists():
        return json.loads(complete.read_text())
    archive=root/'transfer/reference-assets.tar';part=archive.with_suffix('.tar.part')
    url='https://huggingface.co/'+REPOSITORY+'/resolve/'+proof['revision']+'/'+HF_PATH
    session=requests.Session()
    errors=0
    while not archive.exists():
        offset=part.stat().st_size if part.exists() else 0
        if offset==proof['archive_bytes']:
            if sha(part)!=proof['archive_sha256']:
                raise ValueError('Complete target partial failed archive SHA; preserve it')
            os.replace(part,archive);break
        headers={'Authorization':'Bearer '+token}
        if offset:headers['Range']='bytes='+str(offset)+'-'
        try:
            with session.get(url,headers=headers,stream=True,timeout=(8,20)) as response:
                response.raise_for_status()
                if offset and (response.status_code!=206 or not response.headers.get('Content-Range','').startswith('bytes '+str(offset)+'-')):
                    raise ValueError('Resume range not honored')
                with part.open('ab') as stream:
                    for chunk in response.iter_content(4*1024**2):
                        if not chunk:continue
                        if stream.tell()+len(chunk)>proof['archive_bytes']:
                            raise ValueError('Oversized target stream')
                        stream.write(chunk);stream.flush()
                        publish(root/'transfer/progress.json',dict(epoch=time.time(),bytes=stream.tell(),
                            total=proof['archive_bytes'],errors=errors,stage='HF_DOWNLOAD'))
                    os.fsync(stream.fileno())
        except requests.RequestException as error:
            errors+=1
            publish(root/'transfer/failure.json',dict(epoch=time.time(),attempt=errors,
                error_type=type(error).__name__,bytes=part.stat().st_size if part.exists() else 0))
            if errors>=6:raise RuntimeError('Remote direct HF transport failed six bounded attempts') from None
            time.sleep(min(10,errors*2))
    if sha(archive)!=proof['archive_sha256']:
        raise ValueError('Target archive SHA differs')
    manifest=proof['manifest'];expected={row['path']:row for row in manifest['files']}
    with tarfile.open(archive,'r') as tar:
        members=tar.getmembers()
        if {m.name for m in members} != set(expected)|{'manifest.json'} or any(not m.isfile() for m in members):
            raise ValueError('Unexpected archive members')
        for member in members:
            relative=Path(member.name)
            if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe archive name')
            target=assets/relative;target.parent.mkdir(exist_ok=True,parents=True)
            with tar.extractfile(member) as source,target.open('wb') as destination:
                for chunk in iter(lambda:source.read(8*1024**2),b''):destination.write(chunk)
    for name,row in expected.items():
        if (assets/name).stat().st_size!=row['bytes'] or sha(assets/name)!=row['sha256']:
            raise ValueError('Target extracted file SHA differs: '+name)
    result=dict(status='REFERENCE_ASSETS_ALL_TARGET_SHA_VERIFIED',archive_sha256=sha(archive),
        files=len(expected),bytes=proof['archive_bytes'],completed_unix=time.time(),source_revision=proof['revision'])
    publish(complete,result);return result


def main():
    request=json.load(sys.stdin);token=request.pop('token');action=request['action']
    root=Path(UPLOAD_ROOT if action=='upload' else TARGET_ROOT)
    root.mkdir(exist_ok=True,parents=True)
    control=root if action=='upload' else root/'transfer';control.mkdir(exist_ok=True)
    with (control/'transfer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        publish(control/'controller.json',dict(pid=os.getpid(),action=action,started_unix=time.time()))
        try:
            result=upload(token,root) if action=='upload' else download(token,root,request['proof'])
            print(json.dumps({k:v for k,v in result.items() if k!='manifest'}),flush=True)
        except Exception as error:
            publish(control/'worker-failure.json',dict(epoch=time.time(),type=type(error).__name__,
                message=str(error) if isinstance(error,ValueError) else 'Remote transfer worker failed; see typed phase receipt'))
            raise SystemExit(1)


if __name__=='__main__':main()
