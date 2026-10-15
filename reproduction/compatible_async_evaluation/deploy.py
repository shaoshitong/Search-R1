"""Publish-verified remote CODE ONLY deployment; never prepare or run evaluation.

No checkpoint, model, data, environment, log or trajectory is copied. Existing
namespaces return their matching receipt, or refuse replay if still incomplete.
"""
import argparse
import base64
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import zlib

SOURCE_COMMIT='cda611600853bd18a6c259598aa1626da2ae1d80'
SOURCE_BASE=Path('/mnt/data/shaoshitong/search-g1-qprobe3-warmup10-cda6116-20261015')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def once(path,value):
    with Path(path).open('x') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())


def safe_name(name):
    p=Path(name)
    if not name or p.is_absolute() or '..' in p.parts or '.git' in p.parts:
        raise ValueError('CODE_DEPLOY_INVALID_GIT_PATH')
    return p


def verify(root,files):
    for name,sha in files.items():
        p=root/safe_name(name)
        if not p.is_file() or p.is_symlink() or digest(p)!=sha:
            raise ValueError('CODE_DEPLOY_FULL_SHA_MISMATCH:'+name)


def deploy_remote(meta):
    commit=meta['commit']
    if not re.fullmatch('[0-9a-f]{40}',commit):raise ValueError('CODE_DEPLOY_INVALID_COMMIT')
    root=Path('/mnt/data/shaoshitong/search-g1-compatible-eval-'+commit[:7]+'-20261015')
    intent=root/'deploy-intent.json';receipt=root/'deploy-receipt.json'
    if root.exists():
        if not intent.is_file() or not receipt.is_file():
            raise ValueError('CODE_DEPLOY_OCCUPIED_OR_INCOMPLETE_INSPECT_DO_NOT_REPLAY')
        old=json.loads(receipt.read_text())
        if old['source_commit']!=commit or old['source_files_sha256']!=meta['files']:
            raise ValueError('CODE_DEPLOY_EXISTING_BINDING_DIFFERS')
        verify(root/'control',meta['files'])
        return dict(existing_receipt=True,receipt=old,full_sha_reverified=True)
    source_files=json.loads((SOURCE_BASE/'preparation/source-files.json').read_text())
    run=json.loads((SOURCE_BASE/'logs/run-manifest.json').read_text())
    if run['source_commit']!=SOURCE_COMMIT or source_files!=meta['parent_files']:
        raise ValueError('CODE_DEPLOY_FROZEN_SOURCE_BINDING_MISMATCH')
    # Only code bytes are read, so this does not hash model or training outputs.
    verify(SOURCE_BASE/'repo',source_files)
    env=(SOURCE_BASE/'env').resolve()
    if not (env/'bin/python').is_file():raise ValueError('CODE_DEPLOY_EXISTING_ENV_MISSING')
    changes=json.loads(zlib.decompress(base64.b64decode(meta['changes'])))
    if set(changes)!={n for n in meta['files'] if source_files.get(n)!=meta['files'][n]}:
        raise ValueError('CODE_DEPLOY_PATCH_MAP_MISMATCH')
    for name,payload in changes.items():
        safe_name(name)
        if hashlib.sha256(base64.b64decode(payload)).hexdigest()!=meta['files'][name]:
            raise ValueError('CODE_DEPLOY_PATCH_FULL_SHA_MISMATCH:'+name)
    root.mkdir();control=root/'control';control.mkdir()
    once(intent,dict(role='compatible_async_evaluation_code_only_deployment',source_commit=commit,
        frozen_source_commit=SOURCE_COMMIT,frozen_source_base=str(SOURCE_BASE),
        evaluation_prepared=False,gpu_started=False,training_modified=False))
    # Copy only enumerated Git blobs, excluding any untracked runtime products.
    for name in meta['files']:
        target=control/safe_name(name);target.parent.mkdir(parents=True,exist_ok=True)
        if name in changes:target.write_bytes(base64.b64decode(changes[name]))
        else:shutil.copyfile(SOURCE_BASE/'repo'/name,target)
    verify(control,meta['files'])
    once(root/'publication-proof.json',meta['publication'])
    once(root/'code-only-configuration.json',dict(control_repo=str(control),
        environment_root=str(env),environment_copied=False,evaluation_prepared=False,
        checkpoint_bound=False,gpu_started=False,training_modified=False))
    result=dict(role='compatible_async_evaluation_code_only_deployment',source_commit=commit,
        frozen_source_commit=SOURCE_COMMIT,base=str(root),control_repo=str(control),
        source_files_sha256=meta['files'],verified_files=len(meta['files']),
        all_target_source_full_sha_match=True,environment_root=str(env),
        evaluation_prepared=False,checkpoint_bound=False,gpu_started=False,
        training_modified=False,checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    once(receipt,result)
    return result


def deploy(proof_path,control_path,evidence):
    repo=Path(__file__).resolve().parents[2]
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    if subprocess.check_output(['git','status','--porcelain'],cwd=repo):
        raise ValueError('CODE_DEPLOY_REQUIRES_CLEAN_PUBLISHED_TREE')
    proof=json.loads(Path(proof_path).read_text())
    if proof['published_control_commit']!=commit or proof['all_blobs_full_sha_match'] is not True:
        raise ValueError('CODE_DEPLOY_INDEPENDENT_PUBLICATION_PROOF_REQUIRED')
    def mapping(ref):
        result={}
        listing=subprocess.check_output(['git','ls-tree','-r','--format=%(objectname) %(path)',ref],cwd=repo,text=True)
        for row in listing.splitlines():
            oid,name=row.split(' ',1);blob=subprocess.check_output(['git','cat-file','blob',oid],cwd=repo)
            result[name]=hashlib.sha256(blob).hexdigest()
        return result
    parent=mapping(SOURCE_COMMIT);files=mapping(commit)
    if proof['files']!=files:raise ValueError('CODE_DEPLOY_LOCAL_GIT_DIFFERS_FROM_PUBLICATION')
    changes={name:base64.b64encode(subprocess.check_output(['git','show',commit+':'+name],cwd=repo)).decode()
        for name in files if parent.get(name)!=files[name]}
    meta=dict(commit=commit,files=files,parent_files=parent,publication=proof,
        changes=base64.b64encode(zlib.compress(json.dumps(changes).encode())).decode())
    payload='import json\nnamespace={"__name__":"code_only_deploy"}\nexec(compile('+repr(Path(__file__).read_text())+',"code_only_deploy.py","exec"),namespace)\nresult=namespace["deploy_remote"]('+repr(meta)+')\nprint(json.dumps({k:v for k,v in result.items() if k not in ("source_files_sha256","receipt")}))\n'
    from reproduction.h1008_qprobe.management_transport import remote
    return remote(payload,evidence,control_path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--proof',required=True);p.add_argument('--control-path',required=True);p.add_argument('--evidence',required=True)
    args=p.parse_args();print(deploy(args.proof,args.control_path,args.evidence))

if __name__=='__main__':main()
