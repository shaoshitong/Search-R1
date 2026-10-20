"""Stage published code and a qualified native continuation on retained18."""
import argparse
import base64
import gzip
import hashlib
import json
import subprocess
import time
from .control import REPO, OUT, remote


SCRIPT = r'''
import pathlib,json,gzip,base64,hashlib,os,subprocess,time
physical=pathlib.Path('/mnt/data/shaoshitong');canonical='/mnt/data/sihe-share5/workspace/training_data_node_10/shaoshitong/'
source=physical/SOURCE;base=physical/BASE
proof=json.loads((source/('preparation/recovery%d-cpu/full-state-proof.json'%OUTER)).read_text())
assert proof['outer']==OUTER and proof['status']=='DIVERSE_NATIVE_FULL_STATE_CPU_PASS'
assert hashlib.sha256((source/'latest-joint.json').read_bytes()).hexdigest()==proof['joint_sha256']
assert json.loads((source/'logs/cleanup-status.json').read_text())['owned_births_all_gone']
assert not base.exists(),'Reconcile an existing stage; never overwrite'
base.mkdir();(base/'repo').mkdir();(base/'preparation').mkdir();(base/'logs').mkdir()
packet=json.loads(gzip.decompress(base64.b64decode(PACKET)))
for row in packet['manifest']:
 rel=pathlib.Path(row['path']);assert not rel.is_absolute()and'..'not in rel.parts
 target=base/'repo'/rel;target.parent.mkdir(parents=True,exist_ok=True)
 if row['path']in packet['changed']:
  target.write_bytes(base64.b64decode(packet['changed'][row['path']]))
 else:
  original=source/'repo'/rel
  assert original.is_file()and not original.is_symlink()
  assert hashlib.sha256(original.read_bytes()).hexdigest()==row['sha256'],str(rel)
  os.link(original,target)
 assert hashlib.sha256(target.read_bytes()).hexdigest()==row['sha256'],str(rel)
(base/'preparation/source-full-sha.json').write_text(json.dumps(dict(commit=COMMIT,files=packet['manifest'],checked_unix=time.time())))
cmd=['unshare','--mount','--propagation','private','bash',str(base/'repo/reproduction/operations/b_replica_retained18/private_shared_paths.sh'),'env','CUDA_VISIBLE_DEVICES=','SEARCH_G1_PHYSICAL_WORLD_SIZE='+str(WORLD),'SEARCH_G1_B_PREFIX_TOKENS=80','OMP_NUM_THREADS=2','MKL_NUM_THREADS=2','PYTHONDONTWRITEBYTECODE=1','PYTHONPATH='+canonical+BASE+'/repo','/dataset/shaoshitong/Search-G1-common8/env/bin/python','-u','-m',PREPARE_MODULE,'--source',canonical+SOURCE,'--base',canonical+BASE,'--outer',str(OUTER),'--commit',COMMIT]
with(base/'logs/native-preparation.log').open('xb')as log:
 proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
st=pathlib.Path('/proc',str(proc.pid),'stat').read_text().rsplit(')',1)[1].split()
receipt=dict(pid=proc.pid,start_ticks=int(st[19]),command=cmd,commit=COMMIT,created_unix=time.time(),files=len(packet['manifest']),training_started=False)
(base/'preparation/stage-receipt.json').write_text(json.dumps(receipt));print(json.dumps(receipt))
'''


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True)
    p.add_argument('--four',action='store_true')
    p.add_argument('--reuse-four',action='store_true')
    p.add_argument('--base', required=True)
    p.add_argument('--outer', required=True, type=int)
    p.add_argument('--source-commit', required=True)
    a = p.parse_args()
    for value in (a.source, a.base):
        if not value.startswith('search-g1-') or '/' in value or '..' in value:
            raise ValueError('Retained18 task directory name required')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO):
        raise ValueError('Commit and push code before staging')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=REPO, text=True).strip()
    published = subprocess.check_output(['git', 'ls-remote', 'origin', 'refs/heads/'+branch], cwd=REPO, text=True)
    if published.split()[0] != commit:
        raise ValueError('Latest source has not been pushed')
    changed = set(subprocess.check_output(['git', 'diff', '--name-only', a.source_commit, commit], cwd=REPO, text=True).splitlines())
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=REPO).decode().strip('\0').split('\0')
    packet = dict(manifest=[], changed={})
    for name in tracked:
        data = (REPO/name).read_bytes()
        packet['manifest'].append(dict(path=name, sha256=hashlib.sha256(data).hexdigest()))
        if name in changed:
            packet['changed'][name] = base64.b64encode(data).decode()
    header = '\n'.join(k+'='+repr(v) for k,v in dict(SOURCE=a.source, BASE=a.base, OUTER=a.outer,
        COMMIT=commit, WORLD=4 if a.four else 8, PREPARE_MODULE='reproduction.operations.b_replica_retained18.repair_four_prepared'if a.reuse_four else'reproduction.operations.b_replica_retained18.prepare_four_latest'if a.four else'reproduction.operations.b_replica_retained18.prepare_shared_latest', PACKET=base64.b64encode(gzip.compress(json.dumps(packet).encode())).decode()).items())
    evidence = OUT/('stage-qualified-%d-transport.json'%time.time_ns())
    value = json.loads(remote(header+'\n'+SCRIPT, evidence, target='h100-10', established_controller=True))
    (OUT/('CP%d_STAGE_RECEIPT.json'%a.outer)).write_text(json.dumps(value,indent=2))
    print(json.dumps(value),flush=True)


if __name__ == '__main__':
    main()
