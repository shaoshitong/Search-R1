"""Code-only archive deployment; password is read from private memory into RAM."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--control-root',required=True)
    parser.add_argument('--postprocess-only',action='store_true');args=parser.parse_args()
    control=Path(args.control_root);repo=Path(__file__).resolve().parents[3]
    assert (repo/'.git').exists()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo).strip(),'Publish clean source before deployment'
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/qwen7b-probe-warmup-20261019'],cwd=repo,text=True).split()[0]
    assert remote==commit
    archive=subprocess.check_output(['git','archive','--format=tar.gz',commit],cwd=repo)
    sha=hashlib.sha256(archive).hexdigest()
    files={str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.iterdir() if p.is_file()}
    publication=dict(commit=commit,remote_sha=remote,branch='codex/qwen7b-probe-warmup-20261019',archive_sha256=sha,files=files)
    stage='/backup/shaoshitong/qwen7b-native-warmup-code-20261019/source-'+commit+'.tar.gz'
    base='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019'
    archive_target=base+'/preparation/'+('source-postprocess-'+commit[:12]+'.tar.gz' if args.postprocess_only else 'source.tar.gz')
    code='import pathlib,sys,hashlib; p=pathlib.Path('+repr(stage)+');p.parent.mkdir(parents=True,exist_ok=True);d=sys.stdin.buffer.read();assert hashlib.sha256(d).hexdigest()=='+repr(sha)+';p.write_bytes(d)'
    ssh=['ssh','-T','-oConnectTimeout=8','-oBatchMode=yes','hkust3']
    result=subprocess.run(ssh+['python3 -c '+shlex.quote(code)],input=archive,capture_output=True,timeout=45)
    assert result.returncode==0,result.stderr.decode()
    setup='base='+repr(base)+'\npublication='+repr(publication)+'\n'+'''
import pathlib,tarfile,hashlib,json,subprocess
b=pathlib.Path(base);p=b/'preparation/source.tar.gz'
assert hashlib.sha256(p.read_bytes()).hexdigest()==publication['archive_sha256']
repo=b/'repo'
if (b/'config.json').exists():
 if (b/'control/sampling-submit-intent.json').exists():
  repair=json.loads((b/'control/faiss-initialization-failure-recovery.json').read_text())
  assert repair['sealed_batches']==0 and not list((b/'sampling').glob('shard-*/batches/*/complete.json'))
  assert not subprocess.check_output(['squeue','-h','-j',repair['job_id']],universal_newlines=True).strip(),'Never replace active sampling source'
 assert not (b/'control/fit-submit-intent.json').exists(),'Never replace submitted fitting source'
 controller=b/'control/controller.json'
 if controller.exists():
  c=json.loads(controller.read_text());proc=pathlib.Path('/proc')/str(c['pid'])/'stat'
  assert not proc.exists() or proc.read_text().rsplit(')',1)[1].split()[19]!=c['start_ticks'],'Retire owned waiting coordinator before source update'
repo.mkdir(exist_ok=True)
with tarfile.open(str(p),'r:gz') as t:
 for m in t.getmembers():assert not m.name.startswith('/') and '..' not in pathlib.Path(m.name).parts
 t.extractall(str(repo))
for name,sha in publication['files'].items():assert hashlib.sha256((repo/name).read_bytes()).hexdigest()==sha,name
(b/'preparation/source-publication.json').write_text(json.dumps(publication,indent=2))
if (b/'config.json').exists():
 p=b/'config.json';c=json.loads(p.read_text());old=c['source_commit'];c['source_commit']=publication['commit']
 if (b/'control/faiss-initialization-failure-recovery.json').exists():
  repair=json.loads((b/'control/faiss-initialization-failure-recovery.json').read_text())
  history=b/'control'/('retired-'+repair['job_id']);history.mkdir(exist_ok=True)
  for name in ['sampling-submit-intent.json','sampling-submission.json']:
   source=b/'control'/name
   if source.exists():source.rename(history/name)
  for root in (b/'sampling').glob('shard-*'):
   for name in ['failure.json','controller.json']:
    source=root/name
    if source.exists():source.rename(root/(name+'.retired-'+repair['job_id']))
  c['sampling_partition']='acd_ue'
 p.write_text(json.dumps(c,indent=2))
 (b/'preparation/source-upgrade-before-sampling.json').write_text(json.dumps(dict(previous_commit=old,new_commit=publication['commit'],accepted_sampling_batches=0,run_id=c['run_id'])))
for name in ['logs','probe','tmp','cache']:(b/name).mkdir(exist_ok=True)
print(json.dumps(dict(status='SOURCE_DEPLOYED_ALL_FILES_SHA_MATCH',commit=publication['commit'],files=len(publication['files']))))
'''
    if args.postprocess_only:
        setup='base='+repr(base)+'\npublication='+repr(publication)+'\narchive_target='+repr(archive_target)+'\n'+'''
import pathlib,tarfile,hashlib,json
b=pathlib.Path(base);p=pathlib.Path(archive_target)
assert not (b/'control/fit-submit-intent.json').exists(),'Do not alter a submitted fitting runtime'
assert hashlib.sha256(p.read_bytes()).hexdigest()==publication['archive_sha256']
repo=b/('repo-postprocess-'+publication['commit'][:12])
if not repo.exists():
 repo.mkdir()
 with tarfile.open(str(p),'r:gz') as t:
  for m in t.getmembers():assert not m.name.startswith('/') and '..' not in pathlib.Path(m.name).parts
  t.extractall(str(repo))
for name,sha in publication['files'].items():assert hashlib.sha256((repo/name).read_bytes()).hexdigest()==sha,name
publication.update(code_root=str(repo),role='future_merge_fit_and_readonly_audit_only',sampling_source_unchanged=True)
(b/'preparation/postprocess-source-publication.json').write_text(json.dumps(publication,indent=2))
print(json.dumps(dict(status='IMMUTABLE_POSTPROCESS_SOURCE_DEPLOYED',code_root=str(repo),commit=publication['commit'],files=len(publication['files']),sampling_source_unchanged=True)))
'''
    transfer='stage='+repr(stage)+'\nbase='+repr(base)+'\narchive_target='+repr(archive_target)+'\nsetup='+repr(setup)+'\n'+'''
import os,sys,subprocess,shlex
pw=sys.stdin.readline().strip();assert pw;env=dict(os.environ,SSHPASS=pw);pw=None
ssh=['sshpass','-e','ssh','-T','-oStrictHostKeyChecking=yes','-oHostKeyAlias=hpc3login.hpc.hkust-gz.edu.cn','-oUserKnownHostsFile=/backup/shaoshitong/topd-hf-migration-20261009/hpc3-known-hosts','-oConnectTimeout=10','mou176@10.120.81.30']
subprocess.run(ssh+['mkdir -p '+shlex.quote(base+'/preparation')],env=env,check=True)
subprocess.run(['rsync','--checksum','-e',shlex.join(ssh[:-1]),stage,'mou176@10.120.81.30:'+archive_target],env=env,check=True)
subprocess.run(ssh+['python3 -c '+shlex.quote(setup)],env=env,check=True)
'''
    sys.path.insert(0,str(control/'work/prefix-schedule-cp50-20261018'))
    from reproduction.hpc3_prefix5.direct_control import password_for
    result=subprocess.run(ssh+['python3 -c '+shlex.quote(transfer)],input=password_for('mou176')+b'\n',capture_output=True,timeout=60)
    print(result.stdout.decode());assert result.returncode==0,result.stderr.decode()[-1000:]
    destination=control/'outputs/qwen7b-warmup-20261019'/('postprocess-source-publication.json' if args.postprocess_only else 'source-publication.json')
    destination.write_text(json.dumps(publication,indent=2))


if __name__=='__main__':main()
