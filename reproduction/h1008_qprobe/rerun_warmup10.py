"""Once-only scoped warmup10 fresh200 replacement controls; no deletion phase.

Deploy may run locally via the existing management socket. Other phases run on
h100-8 in the new published checkout. Existing intents are never replayed.
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

OLD = Path('/mnt/data/shaoshitong/search-g1-qprobe3-resume5-fb68e88-20261015')
STAGED = Path('/mnt/data/shaoshitong/search-g1-qprobe3-stdloss-545d4cf-20261015-v2')
JOB = 3400
RUN = 'c025948ebc0947f69d86af84871a9237'
SOURCE = 'fb68e88b076917b2684f0fbf72d79a953c527f32'
WARMUP = 'search_g1/qprobe_warmup.py'
TERMINAL = ('CANCELLED', 'FAILED', 'COMPLETED', 'TIMEOUT', 'NODE_FAIL', 'OUT_OF_MEMORY', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE')


def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(path):return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8388608),b''):h.update(block)
    return h.hexdigest()
def once(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.flush();os.fsync(stream.fileno())
def live(row):
    try:return Path('/proc',str(row['pid']),'stat').read_text().rsplit(')',1)[1].split()[19]==str(row['start_ticks'])
    except FileNotFoundError:return False

def authority():
    return dict(observed_utc=utc(),original_message_time='unknown',
        quoted_instructions=['官方权重重新开始200轮','之前那些训练的actor没用...直接删掉'],
        scope='Stop only current3400; fresh official actor200 after10epoch probe warmup on5120 frozen teacher questions; goal remains active; preserve foreign tasks',
        deletion_executed=False,deletion_requires_separate_parent_path_verification=True)

def old_identity():
    manifest=read(OLD/'logs/run-manifest.json');config=read(OLD/'preparation/h1008-launch-config.json')
    assert manifest['run_id']==RUN and manifest['source_commit']==SOURCE
    assert config['source_commit']==SOURCE and config['base']==str(OLD)
    return manifest,config

def validate(base):
    base=Path(base).resolve();assert re.fullmatch(r'/mnt/data/shaoshitong/search-g1-qprobe3-warmup10-[0-9a-f]{7}-20261015',str(base))
    old_identity();config=read(base/'preparation/h1008-launch-config.json')
    assert config['base']==str(base) and config['superseded_job_id']==JOB
    assert config['superseded_training_base']==str(OLD) and config['superseded_run_id']==RUN
    assert config['role']=='fresh_h1008_transformer3_question_probe200' and config.get('restore_receipt') is None
    assert config['warmup_epochs']==10 and config.get('warmup_compatible_source_sha256',{}).keys()=={WARMUP}
    assert not any(config.get(key) for key in ('restore_source_job_id','restore_terminal_relative'))
    for name,digest in read(base/'preparation/source-files.json').items():assert sha(base/'repo'/name)==digest,name
    return config

def deploy_remote(meta):
    old_identity();base=Path(meta['base'])
    assert not base.exists(),'Deployment exists; inspect existing intent/receipt, never replay'
    assert re.fullmatch(r'/mnt/data/shaoshitong/search-g1-qprobe3-warmup10-[0-9a-f]{7}-20261015',str(base))
    original=read(STAGED/'preparation/source-files.json')
    for name,digest in original.items():assert sha(STAGED/'repo'/name)==digest,name
    base.mkdir()
    for name in ('data','preparation','logs','tmp','cache','artifacts','checkpoints'):(base/name).mkdir()
    once(base/'preparation/deploy-intent.json',dict(source_commit=meta['commit'],source_job=3399,
        superseded_job=JOB,official_fresh=True,inherited_trained_actor=False,authority=authority()))
    shutil.copytree(STAGED/'repo',base/'repo',ignore=shutil.ignore_patterns('__pycache__','.git'))
    for name in meta['deleted']:
        path=base/'repo'/name;assert path.resolve().is_relative_to(base/'repo');path.unlink()
    for name,text in json.loads(zlib.decompress(base64.b64decode(meta['changes']))).items():
        path=base/'repo'/name;assert path.resolve().is_relative_to(base/'repo')
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
    for name,digest in meta['files'].items():assert sha(base/'repo'/name)==digest,name
    for name in ('env','bootstrap-tools','bootstrap-assets','assets','retriever-gpu-lib'):
        (base/name).symlink_to((STAGED/name).resolve(),target_is_directory=True)
    for name in ('manifest.json','train.parquet','validation.parquet','stage.parquet','final.parquet','exclusions-audit.json'):
        (base/'data'/name).symlink_to((STAGED/'data'/name).resolve())
    for name in ('official-model-manifest-original.json','predecessor-exit0.json','predecessor-teardown-closure.json'):
        (base/'preparation'/name).write_bytes((STAGED/'preparation'/name).read_bytes())
    config=read(STAGED/'preparation/h1008-launch-config.json')
    cache=Path(config['frozen_teacher_cache']).resolve()
    # Teacher identity is immutable; allow precisely the metrics-only warmup
    # source change, not a different sampler, template, labels or scorer.
    teacher=read(cache/'identity.json')
    assert WARMUP in teacher['rollout_sources'],'Teacher identity schema must be inspected; do not guess compatibility SHA'
    config.update(role='fresh_h1008_transformer3_question_probe200',base=str(base),source_commit=meta['commit'],
        restore_receipt=None,warmup_epochs=10,
        warmup_compatible_source_sha256={WARMUP:teacher['rollout_sources'][WARMUP]},
        superseded_job_id=JOB,superseded_training_base=str(OLD),superseded_run_id=RUN,
        official_manifest=str(base/'preparation/official-model-manifest-original.json'))
    config.pop('restore_source_job_id',None);config.pop('restore_terminal_relative',None)
    (base/'question-features').symlink_to(Path(config['precomputed_question_features']['path']).resolve(),target_is_directory=True)
    once(base/'preparation/source-files.json',meta['files']);once(base/'preparation/h1008-launch-config.json',config)
    validate(base)
    result=dict(base=str(base),source_commit=meta['commit'],verified_files=len(meta['files']),
        all_target_source_full_sha_match=True,warmup_epochs=10,official_fresh_actor=True,
        inherited_trained_actor=False,submitted=False,deleted_actor_paths=[])
    once(base/'preparation/deploy-receipt.json',result);return result

def request_stop(base):
    validate(base);path=base/'preparation/warmup10-stop-intent.json'
    if path.exists():return dict(existing_intent=True,job_id=JOB,action='Inspect same3400; stop is not replayed')
    native=subprocess.check_output(['scontrol','show','job','-o',str(JOB)],text=True)
    fields=dict(re.findall(r'(\w+)=(\S+)',native))
    assert fields['JobId']==str(JOB) and fields['JobState']=='RUNNING' and fields['NodeList']=='h100-8'
    assert str(OLD) in fields['Command'] and str(OLD) in fields['WorkDir']
    ledger=read(OLD/'logs/process-ledger.json')
    births=[row for job in ledger['jobs'].values() for row in job['ledger'].values()]
    assert births and any(live(row) for row in births),'No authenticated original owned process remains'
    logs=[dict(path=str(p),inode=p.stat().st_ino,device=p.stat().st_dev)
        for p in (OLD/'logs').rglob('*') if p.is_file() and not p.is_symlink()]
    once(path,dict(job_id=JOB,run_id=RUN,native_before=native,authority=authority(),
        owned_births=births,original_log_inodes=logs,requested_utc=utc()))
    result=subprocess.run(['scancel','--signal=TERM','--batch',str(JOB)],capture_output=True,text=True,timeout=30)
    receipt=dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,observed_utc=utc())
    once(base/'preparation/warmup10-stop-receipt.json',receipt)
    assert result.returncode==0,result.stderr
    return dict(stop_requested=True,job_id=JOB,goal_paused=False,foreign_processes_untouched=True)

def capture_terminal(base):
    validate(base);path=base/'preparation/warmup10-old-terminal.json'
    if path.exists():return read(path)
    intent=read(base/'preparation/warmup10-stop-intent.json')
    births=intent['owned_births']
    current=read(OLD/'logs/process-ledger.json')
    births+= [row for job in current['jobs'].values() for row in job['ledger'].values()]
    births=list({(row['pid'],row['start_ticks']):row for row in births}.values())
    assert not any(live(row) for row in births),'Owned process births still live'
    result=subprocess.run(['sacct','-j',str(JOB),'-n','-P','-o','JobIDRaw,State,ExitCode'],capture_output=True,text=True)
    rows=[r.split('|') for r in result.stdout.splitlines() if r.split('|')[0]==str(JOB)] if result.returncode==0 else []
    if not rows:
        native=subprocess.check_output(['scontrol','show','job','-o',str(JOB)],text=True)
        fields=dict(re.findall(r'(\w+)=(\S+)',native));assert fields['JobId']==str(JOB) and fields['BatchHost']=='h100-8'
        rows=[[str(JOB),fields['JobState'],fields['ExitCode']]]
    assert len(rows)==1 and rows[0][1].split()[0] in TERMINAL,rows
    assert not list(Path('/sys/fs/cgroup').glob('**/job_'+str(JOB))),'Owned cgroup remains'
    gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True)
    assert not any(int(row.split(',')[0]) in {x['pid'] for x in births} for row in gpu.splitlines() if row.strip())
    hashes=[]
    for row in intent['original_log_inodes']:
        p=Path(row['path']);st=p.stat();same=st.st_ino==row['inode'] and st.st_dev==row['device']
        if p.suffix in ('.log','.out','.err'):assert same,str(p)
        hashes.append(dict(row,terminal_inode=st.st_ino,terminal_device=st.st_dev,
            same_original_inode=same,bytes=st.st_size,sha256=sha(p)))
    once(base/'preparation/warmup10-original-terminal-log-shas.json',hashes)
    cleanup=read(OLD/'logs/cleanup-status.json');assert cleanup['cleaned'] is True
    result=dict(job_id=JOB,run_id=RUN,native_state=rows[0][1],native_exit_code=rows[0][2],
        original_owned_birth_count=len(births),original_births_live=0,cgroup_released=True,
        owned_gpu_contexts_released=True,cleanup_true=True,
        original_log_hash_manifest_sha256=sha(base/'preparation/warmup10-original-terminal-log-shas.json'),
        actor_updates_inherited=False,actor_paths_deleted=[],observed_utc=utc())
    once(path,result);return result

def submit_fresh(base):
    validate(base);terminal=capture_terminal(base)
    assert terminal['job_id']==JOB and terminal['run_id']==RUN and terminal['cleanup_true'] and terminal['cgroup_released']
    for name in ('submit-intent.json','submit-receipt.json'):
        if (base/'preparation'/name).exists():
            return dict(existing_submission=True,receipt=read(base/'preparation/submit-receipt.json')
                if (base/'preparation/submit-receipt.json').exists() else None)
    assert not (base/'logs/run-manifest.json').exists()
    from reproduction.h1008_qprobe.submit import submit
    return submit(base,read(base/'preparation/source-files.json'))

def deploy(proof_path,control_path,evidence):
    repo=Path(__file__).resolve().parents[2]
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    proof=read(proof_path);assert proof['published_control_commit']==commit and proof['all_blobs_full_sha_match']
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo)
    files=proof['files'];names=subprocess.check_output(['git','diff','--name-only','545d4cf',commit],cwd=repo,text=True).splitlines()
    changes={name:(repo/name).read_text() for name in names if name in files}
    meta=dict(commit=commit,files=files,deleted=[name for name in names if name not in files],
        changes=base64.b64encode(zlib.compress(json.dumps(changes).encode())).decode(),
        base='/mnt/data/shaoshitong/search-g1-qprobe3-warmup10-'+commit[:7]+'-20261015')
    source=Path(__file__).read_text()
    script='namespace={"__name__":"warmup10_control"};exec(compile('+repr(source)+',"warmup10_control.py","exec"),namespace)\nprint(json.dumps(namespace["deploy_remote"]('+repr(meta)+')))'
    # Script's final print uses its outer json import, no global SSH changes.
    from reproduction.h1008_qprobe.management_transport import remote
    return remote('import json\n'+script,evidence,control_path)

def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['deploy','request-stop','capture-terminal','submit'])
    p.add_argument('--base',type=Path);p.add_argument('--proof');p.add_argument('--control-path');p.add_argument('--evidence');a=p.parse_args()
    if a.phase=='deploy':
        assert a.proof and a.evidence;print(deploy(a.proof,a.control_path,a.evidence));return
    assert a.base; base=a.base.resolve()
    print(json.dumps({'request-stop':request_stop,'capture-terminal':capture_terminal,'submit':submit_fresh}[a.phase](base)))

if __name__=='__main__':main()
