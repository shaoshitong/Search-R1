"""Same-cluster CPU package snapshot; source environment remains read-only."""
import argparse
import importlib.metadata
import pathlib
import hashlib
import json
import shutil
import time


def stage(run_id):
    if len(run_id)!=32 or any(c not in '0123456789abcdef' for c in run_id):
        raise ValueError('Explicit own run identity required')
    root=pathlib.Path('/mnt/data/shaoshitong/search-g1-prefix-diversity-cpu-package-stage-'+run_id)
    root.mkdir(exist_ok=False)
    site=pathlib.Path('/dataset/shaoshitong/Search-G1-common8/env/lib/python3.10/site-packages')
    needed={'joblib','scikit-learn','scipy','threadpoolctl'}
    dist={d.metadata['Name'].lower():d for d in importlib.metadata.distributions(path=[str(site)]) if d.metadata['Name'].lower() in needed}
    if set(dist)!=needed: raise ValueError('Existing CPU reward dependencies are incomplete')
    files={};versions={};started=time.time()
    for name,d in dist.items():
        versions[name]=d.version
        for item in d.files:
            p=pathlib.Path(d.locate_file(item));rel=p.relative_to(site)
            if '..' in rel.parts or '__pycache__' in rel.parts or not p.is_file():continue
            q=root/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
            data=q.read_bytes();files[str(rel)]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
    manifest=dict(source_site=str(site),versions=versions,files=files,bytes=sum(v['bytes'] for v in files.values()),
        elapsed_seconds=time.time()-started,source_environment_modified=False,via_Mac=False)
    p=root/'package-manifest.json';p.write_text(json.dumps(manifest,sort_keys=True))
    return dict(stage=str(root),manifest_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
        versions=versions,file_count=len(files),bytes=manifest['bytes'],elapsed_seconds=manifest['elapsed_seconds'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);args=p.parse_args();print(json.dumps(stage(args.run_id)))
