"""Remote-only historical-reference supplement; never run automatically.

Published CLI: python historical-supplement.py prepare|start|inspect
  --plan historical-supplement-plan.json --control-commit FULL_SHA
Source prepare runs on h100-8; target start/inspect runs on h100-10. No training,
GPU use, parent inventory mutation, or parent writer intervention is permitted.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess

RUN='f77c3245dcfe49a88a5a99b1dec161e6'
BASE=Path('/mnt/data/shaoshitong/search-g1-qprobe3-binary10-c5fc096-20261015')
EXPORT=Path('/mnt/shared/workspace/training_data_node_8')
BACKUPS=Path('/mnt/data/shaoshitong/search-g1-qprobe3-independent-backup-20261015')
PARENT=BACKUPS/(RUN+'-step5-once-v1')
SOURCE_ROOT=BASE/'preparation/independent-backup-global_step_5-historical-supplement-once-v1'
PARENT_INVENTORY=BASE/'preparation/independent-backup-global_step_5-once-v1/inventory.json'
PAIR=BASE/'artifacts/question-probe/completed-pairs/global_step_5/source-receipt.json'
PAIR_SHA='cdad092617b8966f662dd8d0108f49b40457afe2e695881fea90fa6c1ea49ce7'
PARENT_SHA='9a992b9cd35f7adfd24a492c348b50707bea28198ccdf49b3760753e56b8f45c'
WORKER_SHA='f789e3b58c04a204d8bfbd702f3738f29061e53248777315c8b3e4e4d10aa2fa'
RATE=20971520
RESERVE=137438953472
SOURCE_PREPARATION_COMMIT='b1eedbb1a4a5d020a8a5192abc5e151905ae32ca'
SOURCE_INVENTORY_SHA='23f94ee594327263061b461a4523b4fc871d09368330e4f2963b2bb13c48c98a'
SOURCE_PLAN_SHA='1fe291e1250ee68b695c5c19d7c83838f72848b84ab0115b6d10cdf55b1416e7'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(1024*1024),b''):h.update(data)
    return h.hexdigest()


def read(path):
    p=Path(path)
    if p.is_symlink() or not p.is_file():raise ValueError('REGULAR_BOUND_ARTIFACT_REQUIRED:'+str(p))
    return json.loads(p.read_text())


def once(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())
    fd=os.open(Path(path).parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def validate_plan(plan):
    if plan.get('run_id')!=RUN or plan.get('step')!=5 or plan.get('source_pair_sha256')!=PAIR_SHA or plan.get('parent_inventory_sha256')!=PARENT_SHA:
        raise ValueError('SUPPLEMENT_RUN_PAIR_PARENT_BINDING_DIFFERS')
    if plan.get('recommended_target_root')!=str(BACKUPS/(RUN+'-step5-historical-supplement-once-v1')) or plan.get('worker_path')!=str(BACKUPS/'published-tools/04787779e2faabf15bfbf352869748a77efa7f3c/copy-worker.py'):
        raise ValueError('SUPPLEMENT_NAMESPACE_OR_PUBLISHED_WORKER_DIFFERS')
    if plan.get('rate_bytes_per_second')!=RATE or plan.get('reserve_bytes')!=RESERVE:raise ValueError('SUPPLEMENT_COPY_BUDGET_DIFFERS')
    rows=plan['files']
    if len(rows)!=3 or sum(r['bytes'] for r in rows)!=836891 or plan['bytes']!=836891:raise ValueError('EXACT_THREE_HISTORICAL_FILES_REQUIRED')
    if len({r['source_path'] for r in rows})!=3:raise ValueError('DUPLICATE_SUPPLEMENT_PATH')
    for row in rows:
        path=Path(row['source_path'])
        if not path.is_absolute() or not path.is_relative_to(BASE) or '..' in path.parts or row['payload_path']!=str(path).lstrip('/'):
            raise ValueError('SUPPLEMENT_SOURCE_OR_PAYLOAD_ESCAPES')
    return rows


def validate_pair_rows(pair,rows):
    if pair.get('run_id')!=RUN or pair.get('step')!=5 or pair.get('status')!='QPROBE_COMPLETED_PAIR_FULL_SHA_CPU_STATES_PASS':raise ValueError('REAL_CP5_PAIR_REQUIRED')
    history=pair['historical_artifacts']
    records=[history['development']['before'],history['development']['after5']]+history['reward_archives']
    for row in rows:
        matches=[r for r in records if r['path']==row['source_path']]
        if len(matches)!=1 or matches[0]['full_read_sha256']!=row['expected_sha256'] or matches[0]['bytes']!=row['bytes']:
            raise ValueError('SUPPLEMENT_NOT_PINNED_BY_ORIGINAL_PAIR')


def exported(path):return EXPORT/Path(path).relative_to('/mnt/data')


def prepare_source(plan_path,commit):
    if socket.gethostname().split('.')[0]!='h100-8':raise ValueError('SOURCE_H1008_ONLY')
    plan=read(plan_path);rows=validate_plan(plan)
    if SOURCE_ROOT.exists():raise ValueError('SOURCE_OCCUPIED_INSPECT_DO_NOT_REPLAY')
    if commit!=SOURCE_PREPARATION_COMMIT:raise ValueError('FROZEN_B1_SOURCE_PREPARATION_ONLY_NO_REPREPARE')
    if sha(PAIR)!=PAIR_SHA or sha(PARENT_INVENTORY)!=PARENT_SHA:raise ValueError('SOURCE_PAIR_OR_PARENT_INVENTORY_SHA_DIFFERS')
    pair=read(PAIR);parent=read(PARENT_INVENTORY);validate_pair_rows(pair,rows)
    if parent.get('run_id')!=RUN or parent.get('step')!=5 or parent.get('paired_receipt_sha256')!=PAIR_SHA or len(parent['files'])!=1122:
        raise ValueError('ORIGINAL_1122_PARENT_IDENTITY_DIFFERS')
    if {r['source_path'] for r in rows}&{r['source_path'] for r in parent['files']}:raise ValueError('SUPPLEMENT_ALREADY_IN_PARENT')
    for row in rows:
        p=Path(row['source_path'])
        if p.is_symlink() or p.resolve()!=p or p.stat().st_size!=row['bytes'] or sha(p)!=row['expected_sha256']:raise ValueError('SOURCE_HISTORICAL_FULL_SHA_DIFFERS')
    SOURCE_ROOT.mkdir(mode=0o700)
    once(SOURCE_ROOT/'INTENT.json',dict(run_id=RUN,step=5,control_commit=commit,source_pair_sha256=PAIR_SHA,parent_inventory_sha256=PARENT_SHA,plan_sha256=sha(plan_path),copy_launched=False))
    inventory=dict(schema_version=1,status='BACKUP_INVENTORY_ONLY',run_id=RUN,step=5,source_base=str(BASE),paired_receipt_sha256=PAIR_SHA,parent_inventory_sha256=PARENT_SHA,files=rows,bytes=836891,control_commit=commit,semantic_reload_performed=False,all_experiment_backup_complete=False)
    once(SOURCE_ROOT/'inventory.json',inventory)
    receipt=dict(status='SOURCE_HISTORICAL_SUPPLEMENT_INVENTORY_READY',run_id=RUN,step=5,source_pair_sha256=PAIR_SHA,parent_inventory_sha256=PARENT_SHA,inventory_path=str(SOURCE_ROOT/'inventory.json'),inventory_sha256=sha(SOURCE_ROOT/'inventory.json'),files=3,bytes=836891,control_commit=commit,copy_launched=False)
    once(SOURCE_ROOT/'RECEIPT.json',receipt)
    return receipt


def parent_finished(parent):
    result=read(parent/'RESULT.json');intent=read(parent/'INTENT.json');launch=read(parent/'LAUNCH.json')
    if result.get('status')!='INDEPENDENT_INVENTORY_FULL_SHA_PASS' or result.get('run_id')!=RUN or result.get('step')!=5 or result.get('inventory_sha256')!=PARENT_SHA or result.get('files')!=1122 or result.get('bytes')!=137694774279 or result.get('final_directory')!=str(parent/'checkpoint-closure'):
        raise ValueError('PARENT_1122_FULL_SHA_RESULT_REQUIRED')
    if intent.get('inventory_sha256')!=PARENT_SHA or intent.get('run_id')!=RUN or intent.get('worker_source_sha256')!=WORKER_SHA or intent.get('rate_bytes_per_second')!=RATE or intent.get('ordinary_reserve_bytes')!=RESERVE:
        raise ValueError('PARENT_WRITER_BINDING_DIFFERS')
    if launch.get('pid')!=2245796 or str(launch.get('start_ticks'))!='1913194447':raise ValueError('ORIGINAL_PARENT_WRITER_BIRTH_DIFFERS')
    if sha(parent/'copy-worker.py')!=WORKER_SHA or sha(parent/'checkpoint-closure/inventory.json')!=PARENT_SHA:raise ValueError('PARENT_CONTROL_OR_FINAL_INVENTORY_DIFFERS')
    proc=Path('/proc')/str(launch['pid'])/'stat'
    try:
        fields=proc.read_text().rsplit(')',1)[1].split()
        if fields[19]==str(launch['start_ticks']) and fields[0]!='Z':raise ValueError('PARENT_WRITER_STILL_LIVE_NO_PARALLEL_COPY')
    except FileNotFoundError:pass
    return dict(result_sha256=sha(parent/'RESULT.json'),writer_birth_not_live=True)


def validate_prepared_source(receipt,inventory,intent,plan,inventory_sha,plan_sha):
    # Execution-control fixes must not rewrite/reprepare the already consumed b1 source.
    if inventory_sha!=SOURCE_INVENTORY_SHA or plan_sha!=SOURCE_PLAN_SHA:
        raise ValueError('FROZEN_B1_SOURCE_INVENTORY_OR_PLAN_SHA_DIFFERS')
    if receipt.get('status')!='SOURCE_HISTORICAL_SUPPLEMENT_INVENTORY_READY' or receipt.get('run_id')!=RUN or receipt.get('step')!=5 or receipt.get('source_pair_sha256')!=PAIR_SHA or receipt.get('parent_inventory_sha256')!=PARENT_SHA or receipt.get('control_commit')!=SOURCE_PREPARATION_COMMIT or receipt.get('inventory_path')!=str(SOURCE_ROOT/'inventory.json') or receipt.get('inventory_sha256')!=SOURCE_INVENTORY_SHA:
        raise ValueError('SUPPLEMENT_FROZEN_SOURCE_RECEIPT_DIFFERS')
    if inventory.get('status')!='BACKUP_INVENTORY_ONLY' or inventory.get('run_id')!=RUN or inventory.get('step')!=5 or inventory.get('files')!=plan['files'] or inventory.get('bytes')!=836891 or inventory.get('paired_receipt_sha256')!=PAIR_SHA or inventory.get('parent_inventory_sha256')!=PARENT_SHA or inventory.get('control_commit')!=SOURCE_PREPARATION_COMMIT:
        raise ValueError('SUPPLEMENT_FROZEN_INVENTORY_BINDINGS_DIFFER')
    if intent.get('run_id')!=RUN or intent.get('step')!=5 or intent.get('control_commit')!=SOURCE_PREPARATION_COMMIT or intent.get('source_pair_sha256')!=PAIR_SHA or intent.get('parent_inventory_sha256')!=PARENT_SHA or intent.get('plan_sha256')!=SOURCE_PLAN_SHA:
        raise ValueError('SUPPLEMENT_FROZEN_SOURCE_INTENT_DIFFERS')


def require_actual_nfs_mount(source):
    # An autofs mountpoint can produce multiple findmnt rows. Bind the mounted
    # filesystem to the actual file's stat device, never the first/arbitrary row.
    source=Path(source);before=source.stat().st_dev
    device=str(os.major(before))+':'+str(os.minor(before))
    raw=subprocess.check_output(['findmnt','--raw','-n','-T',str(source),'-o','FSTYPE,MAJ:MIN'],text=True)
    matching=[]
    for line in raw.splitlines():
        fields=line.split()
        if len(fields)!=2:raise ValueError('SOURCE_MOUNT_RECORD_INVALID')
        if fields[1]==device:matching.append(fields[0])
    if len(matching)!=1 or matching[0] not in ('nfs','nfs4') or source.stat().st_dev!=before:
        raise ValueError('ACTUAL_SOURCE_DEVICE_NFS_MOUNT_REQUIRED')
    return dict(device=device,filesystem=matching[0])


def start_target(plan_path,commit):
    if socket.gethostname().split('.')[0]!='h100-10':raise ValueError('TARGET_H10010_ONLY')
    plan=read(plan_path);validate_plan(plan);root=Path(plan['recommended_target_root'])
    if root.exists():raise ValueError('TARGET_OCCUPIED_INSPECT_UNKNOWN_OUTCOME_DO_NOT_REPLAY')
    # All checks precede a new copy intent. No mutation of the existing bundle.
    parent=parent_finished(PARENT)
    receipt=read(exported(SOURCE_ROOT/'RECEIPT.json'));inventory=exported(receipt['inventory_path'])
    small=read(inventory);source_intent=read(exported(SOURCE_ROOT/'INTENT.json'))
    validate_prepared_source(receipt,small,source_intent,plan,sha(inventory),sha(plan_path))
    if sha(exported(PAIR))!=PAIR_SHA or sha(exported(PARENT_INVENTORY))!=PARENT_SHA:raise ValueError('EXPORTED_PAIR_OR_PARENT_CHANGED')
    validate_pair_rows(read(exported(PAIR)),plan['files'])
    code=Path(plan['worker_path'])
    if sha(code)!=WORKER_SHA:raise ValueError('PUBLISHED_GENERIC_COPY_WORKER_SHA_DIFFERS')
    spec=importlib.util.spec_from_file_location('published_historical_supplement_copy',code);worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
    worker.validate_environment(root)
    if worker.RATE!=RATE or worker.RESERVE!=RESERVE:raise ValueError('GENERIC_COPY_BUDGET_DIFFERS')
    if shutil.disk_usage(root.parent).free<836891+RESERVE:raise ValueError('SUPPLEMENT_ORDINARY_128GIB_RESERVE_REQUIRED')
    for row in plan['files']:
        source=exported(row['source_path'])
        require_actual_nfs_mount(source)
        if source.is_symlink() or not source.resolve().is_relative_to(EXPORT.resolve()) or source.stat().st_dev==root.parent.stat().st_dev or source.stat().st_size!=row['bytes']:
            raise ValueError('INDEPENDENT_NFS_SOURCE_EXT4_TARGET_REQUIRED')
    # Serialize this once controller; the original worker has already finished.
    with (BACKUPS/'.historical-supplement-launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if root.exists():raise ValueError('TARGET_INTENT_RACE_INSPECT_DO_NOT_REPLAY')
        parent_finished(PARENT)
        launched=worker.launch(root,inventory,receipt['inventory_sha256'],RUN)
        if not (root/'LAUNCH.json').is_file() or 'pid' not in launched:raise ValueError('UNKNOWN_LAUNCH_OUTCOME_INSPECT_NO_REPLAY')
        once(root/'SUPPLEMENT-RECEIPT.json',dict(control_commit=commit,run_id=RUN,step=5,source_pair_sha256=PAIR_SHA,parent_inventory_sha256=PARENT_SHA,parent_release=parent,source_preparation_control_commit=SOURCE_PREPARATION_COMMIT,source_plan_sha256=SOURCE_PLAN_SHA,inventory_sha256=receipt['inventory_sha256'],files=3,bytes=836891,launch=launched,semantic_reload_performed=False,all_experiment_backup_complete=False))
        return dict(root=str(root),launch=launched,files=3,bytes=836891,complete=False)


def inspect(plan_path):
    plan=read(plan_path);validate_plan(plan);host=socket.gethostname().split('.')[0]
    if host=='h100-8':root=SOURCE_ROOT
    elif host=='h100-10':root=Path(plan['recommended_target_root'])
    else:raise ValueError('TASK_SOURCE_OR_TARGET_NODE_ONLY')
    records={}
    for name in ('INTENT.json','RECEIPT.json','LAUNCH.json','SUPPLEMENT-RECEIPT.json','RESULT.json','ERROR.json','PROGRESS.json'):
        p=root/name
        if p.exists():
            try:records[name]=dict(sha256=sha(p),value=read(p))
            except (ValueError,OSError) as error:records[name]=dict(error=repr(error),complete=False)
    return dict(root=str(root),exists=root.exists(),records=records,action='inspection only; never replay')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('prepare','start','inspect'));p.add_argument('--plan',required=True);p.add_argument('--control-commit',required=True);a=p.parse_args()
    if len(a.control_commit)!=40 or any(c not in '0123456789abcdef' for c in a.control_commit):p.error('full published control commit required')
    control=read(Path(__file__).parent/'RECEIPT.json')
    if control.get('commit')!=a.control_commit or control.get('status')!='HISTORICAL_SUPPLEMENT_CODE_ONLY_FULL_SHA_PASS':raise ValueError('PUBLISHED_SUPPLEMENT_CONTROL_RECEIPT_REQUIRED')
    for name,expected in control['files'].items():
        if Path(name).name!=name or sha(Path(__file__).parent/name)!=expected:raise ValueError('PUBLISHED_SUPPLEMENT_CONTROL_SHA_CHANGED')
    if Path(a.plan).resolve()!=Path(__file__).parent/'historical-supplement-plan.json':raise ValueError('PUBLISHED_PLAN_PATH_REQUIRED')
    fn={'prepare':prepare_source,'start':start_target,'inspect':lambda path,commit:inspect(path)}[a.phase]
    print(json.dumps(fn(a.plan,a.control_commit)))

if __name__=='__main__':main()
