"""Verify the complete dataset archive, extract only its two canonical inputs.

Historical rollout/replay files are never materialized by this new experiment.
The compressed archive remains on HKUST2; selected inputs take the normal peer
route and retain their individual source-manifest SHA bindings.
"""
import hashlib
import io
import json
from pathlib import PurePosixPath
import tarfile
import time

try:
    from reproduction.prefix20_joint4 import assets
except ModuleNotFoundError:
    import assets


SELECTED={
    'canonical/train.parquet':(17823593,'5d40180ba54d35513bb780408caa068d7d60123214a49b0739f835a2ae47c8a1'),
    'canonical/eligible-question-pool.json':(41242673,'ae13710254e7ae77cf5531b62a755ec2000bc3fba20051bbafaff1f6714aa8fc'),
}


def unpack_selected(root,archive,manifest):
    records={r['path']:r for r in manifest['files']};seen=set();files=[]
    paths=[root/r['name'] for r in archive['parts']];digest=hashlib.sha256()
    for path in paths:
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    if digest.hexdigest()!=archive['concatenated_sha256']:
        raise ValueError('CANONICAL_FULL_ARCHIVE_SHA_DIFFERS')
    if any((records[name]['bytes'],records[name]['sha256'])!=binding for name,binding in SELECTED.items()):
        raise ValueError('CANONICAL_SELECTED_MEMBER_BINDING_DIFFERS')
    with io.BufferedReader(assets.PartsReader(paths)) as stream,tarfile.open(fileobj=stream,mode='r|gz') as tar:
        for member in tar:
            name=str(PurePosixPath(member.name))
            if member.isdir():continue
            if (not member.isfile() or PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts
                    or name in seen or name not in set(records)|{'MANIFEST.json'}):
                raise ValueError('CANONICAL_UNSAFE_OR_UNEXPECTED_ARCHIVE_MEMBER')
            seen.add(name)
            if name=='MANIFEST.json':
                if json.load(tar.extractfile(member))!=manifest:raise ValueError('CANONICAL_EMBEDDED_MANIFEST_DIFFERS')
                continue
            if name not in SELECTED:continue
            target=root/'selected'/name;target.parent.mkdir(parents=True,exist_ok=True)
            if any(p.is_symlink() for p in [target,*target.parents]):raise ValueError('CANONICAL_TARGET_SYMLINK')
            temporary=target.with_name(target.name+'.extract-part');h=hashlib.sha256();count=0
            with temporary.open('xb') as out:
                source=tar.extractfile(member)
                for block in iter(lambda:source.read(1024*1024),b''):out.write(block);h.update(block);count+=len(block)
                out.flush();__import__('os').fsync(out.fileno())
            if (count,h.hexdigest())!=SELECTED[name]:raise ValueError('CANONICAL_EXTRACTED_MEMBER_SHA_DIFFERS')
            if target.exists():
                if target.stat().st_size!=count or assets.sha(target)!=h.hexdigest():raise ValueError('CANONICAL_EXISTING_MEMBER_DIFFERS')
                temporary.unlink()
            else:temporary.rename(target)
    if set(records)-seen:raise ValueError('CANONICAL_ARCHIVE_MEMBER_SET_INCOMPLETE')
    for name,(size,sha) in SELECTED.items():
        target=root/'selected'/name
        if target.stat().st_size!=size or assets.sha(target)!=sha:raise ValueError('CANONICAL_FRESH_MEMBER_SHA_DIFFERS')
        files.append(dict(path=name,bytes=size,sha256=sha))
    proof=dict(status='CANONICAL_SELECTED_MEMBERS_FRESH_SHA_VERIFIED',snapshot_epoch=68,
        archive_sha256=digest.hexdigest(),full_compressed_archive_verified=True,
        all_archive_members_extracted=False,historical_rollouts_extracted=False,files=files,checked_unix=time.time())
    assets.atomic(root/'canonical-verified.json',proof)
    return proof


if __name__=='__main__':
    assets.unpack=unpack_selected
    try:assets.main()
    except Exception as error:
        print(json.dumps(dict(status='CANONICAL_ASSET_COMPONENT_FAILED',error_type=type(error).__name__)),file=__import__('sys').stderr)
        raise SystemExit(1)
