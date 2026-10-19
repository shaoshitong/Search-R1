"""Read-only frozen CF manifest metadata parity; never reads raw Q/A or generates."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def audit(root, extras=()):
    root=Path(root);mirror=root/'a800-mirror';specs=[]
    for label,prefix in [('Search-G1','search-g1'),('Soft-to-Hard','soft-to-hard'),('OPSD3304','opsd3304')]:
        specs.extend([dict(label=label,protocol='v4',path=str(mirror/(prefix+'-cf-full-v4')/'manifest.json')),
                      dict(label=label,protocol='v5',path=str(mirror/(prefix+'-cf-strict-v5')/'manifest.json'))])
    for label,name in [('TOP-D alpha=0.1','topd-alpha01-v4'),('OPD alpha=1','opd-alpha1-v4')]:
        specs.extend([dict(label=label,protocol='v4',path=str(root/name/'cf-paper-strict-v4/manifest.json')),
                      dict(label=label,protocol='v5',path=str(root/name/'cf-strict-v5/manifest.json'))])
    specs.extend(extras);records=[];missing=[]
    for spec in specs:
        path=Path(spec['path'])
        if not path.is_file():missing.append(spec);continue
        raw=path.read_bytes();m=json.loads(raw);payload=dict(m);self_hash=payload.pop('manifest_sha256',None)
        records.append(dict(label=spec['label'],protocol_group=spec['protocol'],manifest_path=str(path),
            method=m['method'],protocol=m['protocol_version'],actor_version=m.get('actor_version'),seed=m.get('seed'),
            cf_config_sha256=m['cf_config_sha256'],cf_config=m['cf_config'],code_files=m['code_files'],
            checkpoint_hash=m['checkpoint_hash'],normal_source_sha256=m['source']['sha256'],
            manifest_file_sha256=hashlib.sha256(raw).hexdigest(),
            manifest_self_hash_valid=self_hash==digest(payload),config_hash_valid=m['cf_config_sha256']==digest(m['cf_config']),
            code_mapping_binding_valid=m['cf_config'].get('implementation_sha256')==digest(m['code_files'])))
    groups={}
    for protocol in sorted({r['protocol_group']for r in records}):
        rows=[r for r in records if r['protocol_group']==protocol]
        groups[protocol]=dict(method_count=len(rows),methods=[r['label']for r in rows],
            cf_config_identical=len({digest(r['cf_config'])for r in rows})==1,
            cf_config_hash_identical=len({r['cf_config_sha256']for r in rows})==1,
            code_files_mapping_identical=len({digest(r['code_files'])for r in rows})==1,
            protocol_identical=len({r['protocol']for r in rows})==1,
            actor_version_identical=len({r['actor_version']for r in rows})==1,
            seed_identical=len({r['seed']for r in rows})==1,
            checkpoint_hashes_all_distinct=len({r['checkpoint_hash']for r in rows})==len(rows),
            unique_config_hashes=sorted({r['cf_config_sha256']for r in rows}),
            unique_code_mapping_hashes=sorted({digest(r['code_files'])for r in rows}))
    cross={}
    for label in sorted({r['label']for r in records}):
        rows=[r for r in records if r['label']==label];present={r['protocol_group']for r in rows}
        cross[label]=dict(protocol_groups=sorted(present),v4_v5_both_present=present=={'v4','v5'},
            normal_source_sha256_identical=len({r['normal_source_sha256']for r in rows})==1,
            checkpoint_hash_identical=len({r['checkpoint_hash']for r in rows})==1,
            unique_source_sha256=sorted({r['normal_source_sha256']for r in rows}))
    bindings=all(r['manifest_self_hash_valid']and r['config_hash_valid']and r['code_mapping_binding_valid']for r in records)
    parity=all(g['cf_config_identical']and g['cf_config_hash_identical']and g['code_files_mapping_identical']and g['checkpoint_hashes_all_distinct']for g in groups.values())
    same_sources=all(v['v4_v5_both_present']and v['normal_source_sha256_identical']and v['checkpoint_hash_identical']for v in cross.values())
    return dict(checked_at=time.time(),state='MANIFEST_METADATA_PARITY_VERIFIED'if not missing and bindings and parity and same_sources else 'PARITY_DIFFERENCES_OR_MISSING',
        scope='Frozen-manifest metadata equality only; no raw Q/A or model generation; runtime byte verification belongs to separate acceptance receipts',
        raw_question_answer_data_read=False,gpu_used=False,all_manifest_bindings_valid=bindings,missing=missing,
        within_protocol=groups,within_method=cross,records=records)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',required=True)
    p.add_argument('--extra-specs',help='Optional JSON list of {label,protocol,path}; use for explicit Easy paths when published')
    args=p.parse_args();extras=json.loads(Path(args.extra_specs).read_text())if args.extra_specs else []
    print(json.dumps(audit(args.root,extras),ensure_ascii=False,sort_keys=True))


if __name__=='__main__':main()
