"""Versioned retained corpus-cache relocation; no live cache or payload edits.

Reuse existing independent full-content SHA evidence. Correct only a private
receipt's source stat binding, because backup copies legitimately have another
mtime. Native frozen prepare_corpus remains unchanged and rejects mismatches.
"""
import argparse,hashlib,json,os,time
from pathlib import Path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def once(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x')as f:json.dump(value,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def identity(path):
    s=Path(path).stat();return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns]

def plan(retained,retention_sha):
    retained=Path(retained);proof=retained/'cache2-RESULT.json'
    if sha(proof)!=retention_sha:raise ValueError('Previously independently verified retention result changed')
    result=json.loads(proof.read_text())
    if result['status']!='FIXED_INPUT_RETENTION_COMPLETE' or result['live_training_changed']:raise ValueError('Need completed independent retention')
    cache_rel=Path('shaoshitong/search-g1-prefix20-online1000-20261016-ca1c60f/retrieval-cache')
    cache=retained/cache_rel;receipt=cache/'corpus.json';old=json.loads(receipt.read_text())
    relative=Path(old['source']).relative_to('/mnt/data/shaoshitong');archive=retained/'shaoshitong'/relative
    by_suffix={}
    for row in result['files']:
        target=Path(row['target']);parts=target.parts
        # Both authenticated aliases refer to this exact retained namespace.
        marker=retained.name
        if marker not in parts:raise ValueError('Retention payload escaped its recorded root')
        rel=Path(*parts[parts.index(marker)+1:]);by_suffix[rel.as_posix()]=row
    a=by_suffix[('shaoshitong/'+relative.as_posix())]
    if a['kind']!='HARDLINK_TO_EXISTING_IMMUTABLE_FULL_SHA_BACKUP' or identity(archive)!=a['identity']:
        raise ValueError('Sealed immutable archive inode/mtime differs')
    rows=[]
    for rel,row in by_suffix.items():
        path=Path(rel)
        if not path.is_relative_to(cache_rel):continue
        suffix=path.relative_to(cache_rel)
        if suffix.name.endswith('.lock'):raise ValueError('Never migrate a process lock')
        source=retained/path;s=source.stat()
        if source.is_symlink() or row['kind']!='NEW_COPY_SOURCE_TARGET_FULL_SHA_PASS' or row['source_sha256']!=row['destination_sha256'] or s.st_size!=row['bytes'] or s.st_mtime_ns!=row['source_identity'][-1]:
            raise ValueError('Retained cache payload differs from its completed source/target readback')
        rows.append(dict(relative=suffix.as_posix(),path=str(source),bytes=s.st_size,identity=identity(source),prior_full_sha=row['destination_sha256']))
    names={r['relative']:r for r in rows}
    if not {'wiki_dump.jsonl','corpus.json'}<=names.keys() or names['wiki_dump.jsonl']['prior_full_sha']!=old['content_sha256'] or names['wiki_dump.jsonl']['bytes']!=old['content_bytes'] or sha(receipt)!=names['corpus.json']['prior_full_sha']:
        raise ValueError('Original receipt/extracted content differ from the actual independent copy proof')
    current=archive.stat()
    if current.st_size!=old['source_size']:raise ValueError('Archive size changed, not merely location metadata')
    new=dict(old,source_mtime_ns=current.st_mtime_ns)
    key=hashlib.sha256(json.dumps(dict(retention_sha=retention_sha,old_receipt_sha=sha(receipt),archive_identity=identity(archive)),sort_keys=True).encode()).hexdigest()
    return dict(status='RETAINED_CACHE_RELOCATION_PLANNED',root=str(retained/'corpus-cache-relocations'/key),retained=str(retained),retention_result=str(proof),retention_result_sha256=retention_sha,archive=str(archive),archive_identity=identity(archive),archive_prior_full_sha=a['proven_sha256'],virtual_archive=old['source'],virtual_cache='/mnt/data/'+cache_rel.as_posix(),old_receipt=old,new_receipt=new,cache_files=rows,large_payload_hash_repeated=False,live_training_changed=False)

def apply(retained,retention_sha):
    p=plan(retained,retention_sha);root=Path(p['root']);done=root/'RESULT.json'
    if done.exists():return json.loads(done.read_text())
    if (root/'INTENT.json').exists():raise RuntimeError('Existing incomplete relocation intent: inspect, never replay')
    root.mkdir(parents=True,exist_ok=True);once(root/'INTENT.json',p);out=root/'cache';out.mkdir();manifest=[]
    for row in p['cache_files']:
        if row['relative']=='corpus.json':continue
        source=Path(row['path']);target=out/row['relative'];target.parent.mkdir(parents=True,exist_ok=True)
        if identity(source)!=row['identity']:raise ValueError('Retained source changed after planning')
        if row['bytes']>=1024**2:
            os.link(source,target);kind='SAME_INODE_AS_PRIOR_INDEPENDENT_FULL_SHA_PAYLOAD'
            if identity(source)!=identity(target):raise ValueError('Payload hardlink differs')
        else:
            with target.open('xb')as f:f.write(source.read_bytes());f.flush();os.fsync(f.fileno())
            os.utime(target,ns=(source.stat().st_atime_ns,source.stat().st_mtime_ns));kind='PRIVATE_SMALL_METADATA_COPY'
            if sha(target)!=row['prior_full_sha']:raise ValueError('Private metadata copy SHA differs')
        manifest.append(dict(row,target=str(target),kind=kind))
    once(out/'corpus.json',p['new_receipt'])
    result=dict(p,status='RETAINED_CACHE_BINDING_PREPARED_CPU_RUNTIME_VALIDATION_PENDING',created_unix=time.time(),cache=str(out),private_receipt_sha256=sha(out/'corpus.json'),files=manifest,original_cache_receipt_unchanged=True,archive_timestamp_changed=False,semantic_model_reload_performed=False)
    original=Path(p['retained'])/'shaoshitong/search-g1-prefix20-online1000-20261016-ca1c60f/retrieval-cache/corpus.json'
    if json.loads(original.read_text())!=p['old_receipt']:raise ValueError('Original cache was modified')
    once(done,result);return result

def overlay_script(result):
    """Only a new unshared mount namespace can apply this private cache view."""
    import shlex
    bootstrap='/root/.local/share/search-g1/prefix5-retained-bootstrap/6f970c9201962423daf2d463f9135f0f3cc6b16e5df9b9c12be24d5b7dbb3f60/h10011_retained_paths.sh'
    return "#!/bin/bash\nset -euo pipefail\n[[ $(hostname) == h100-11 ]]\n[[ $(readlink /proc/self/ns/mnt) != $(readlink /proc/1/ns/mnt) ]]\n"+"exec bash "+shlex.quote(bootstrap)+" /bin/bash -c "+shlex.quote('set -euo pipefail; mount --bind "$1" "$2"; shift 2; exec "$@"')+" _ "+shlex.quote(result['cache'])+" "+shlex.quote(result['virtual_cache'])+' "$@"\n'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('plan','apply'));p.add_argument('--retained',required=True);p.add_argument('--retention-sha',required=True);a=p.parse_args();print(json.dumps((plan if a.action=='plan'else apply)(a.retained,a.retention_sha)))
