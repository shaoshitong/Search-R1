"""Reuse only authenticated official-zero reward calibration, never live150."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import uuid


def digest(path):
    sha=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):sha.update(block)
    return sha.hexdigest()


def validate_zero(run,source_run,archive):
    if run.get('completed_updates')!=0 or run.get('full_state_restore'):raise ValueError('INITIAL_REWARD_ADOPTION_REQUIRES_FRESH_RUN')
    if run['initial_model']!=run['reference_model']:raise ValueError('FRESH_ACTOR_REFERENCE_DIFFER')
    if source_run['model_revision']!=run['model_revision'] or run['model_revision']!='aa8e72537993ba99e69dfaafa59ed015b17504d1':raise ValueError('REWARD_ZERO_OFFICIAL_REVISION')
    if archive.get('round')!=0 or archive.get('run_id')!=source_run['run_id']:raise ValueError('REWARD_ZERO_ARCHIVE_REQUIRED')
    if str(Path(archive['encoder']).resolve())!=str(Path(source_run['initial_model']).resolve()):raise ValueError('REWARD_ZERO_ENCODER_IS_NOT_INITIAL')
    if not archive.get('encoder_sha256'):raise ValueError('REWARD_ZERO_ENCODER_SHA_MISSING')
    for name,sha in archive['encoder_sha256'].items():
        if run['initial_model_sha256'].get(name)!=sha or source_run['initial_model_sha256'].get(name)!=sha:raise ValueError('REWARD_ZERO_ENCODER_BYTES_DIFFER:'+name)
    for name in ('manifest.json','validation.parquet','calibration-300.parquet','audit.json'):
        if run['input_sha256'].get(name)!=source_run['input_sha256'].get(name):raise ValueError('REWARD_ZERO_CALIBRATION_INPUT_DIFFERS:'+name)
    report=archive.get('calibration_report',{})
    if report.get('unique_question_count')!=300 or report.get('question_grouped_cv') is not True or report.get('require_all_cv_folds') is not True:raise ValueError('REWARD_ZERO_REAL_CALIBRATION_REPORT_MISSING')
    if set(archive['head_sha256'])!={'commit_probe.pkl','boundary_probe.pkl'}:raise ValueError('REWARD_ZERO_HEAD_SET')
    return True


def once(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())


def adopt(base,source,archive_path,expected_archive_sha):
    base=Path(base).resolve();source=Path(source).resolve();path=Path(archive_path).resolve()
    if not path.is_relative_to(source/'artifacts') or path.name!='version-manifest.json':raise ValueError('REWARD_ZERO_ARCHIVE_PATH')
    if digest(path)!=expected_archive_sha:raise ValueError('REWARD_ZERO_ARCHIVE_FULL_SHA')
    run=json.loads((base/'logs/run-manifest.json').read_text());source_run=json.loads((source/'logs/run-manifest.json').read_text());archive=json.loads(path.read_text())
    validate_zero(run,source_run,archive)
    directory=path.parent;files={}
    for sub,key in [('heads','head_sha256'),('dumps','dump_sha256')]:
        for name,sha in archive[key].items():
            if Path(name).name!=name:raise ValueError('REWARD_ZERO_FILENAME')
            original=directory/sub/name
            if original.is_symlink() or digest(original)!=sha:raise ValueError('REWARD_ZERO_FILE_FULL_SHA:'+name)
            files[str(Path(sub)/name)]=sha
    report=directory/'heads/calibration-report.json'
    if digest(report)!=archive['calibration_report_sha256']:raise ValueError('REWARD_ZERO_REPORT_FULL_SHA')
    files['heads/calibration-report.json']=digest(report)
    target=base/'artifacts'/('round_0-reused-'+uuid.uuid4().hex)
    once(base/'logs/initial-zero-adoption-intent.json',dict(run_id=run['run_id'],source=str(source),archive=str(path),archive_sha256=expected_archive_sha,target=str(target)))
    target.mkdir();(target/'heads').mkdir();(target/'dumps').mkdir()
    for name,sha in files.items():
        shutil.copyfile(directory/name,target/name)
        if digest(target/name)!=sha:raise ValueError('REWARD_ZERO_TARGET_FULL_SHA:'+name)
    live=copy.deepcopy(archive)
    live.update(run_id=run['run_id'],version=run['run_id']+':0:'+uuid.uuid4().hex,
        encoder=run['initial_model'],commit=str(target/'heads/commit_probe.pkl'),boundary=str(target/'heads/boundary_probe.pkl'),
        adoption_origin=dict(kind='h1008_official_zero_archive_v1',source_run_id=source_run['run_id'],archive=str(path),archive_sha256=expected_archive_sha,
            historical_official_zero_calibration_reused=True,new_calibration_generated=False,actor_optimizer_scheduler_rng_inherited=False),published_unix=time.time())
    once(target/'version-manifest.json',live);(base/'artifacts/live').mkdir();once(base/'artifacts/live/manifest.json',live)
    receipt=dict(run_id=run['run_id'],version=live['version'],round=0,source_archive=str(path),source_archive_sha256=expected_archive_sha,
        source_target_files_full_sha=files,actual_new_probe_endpoint_admitted=False,actual_training_started=False)
    once(base/'logs/initial-zero-adoption-receipt.json',receipt)
    return receipt


def verify_adopted(base,run,live,archive_path):
    base=Path(base).resolve();archive_path=Path(archive_path).resolve();origin=live['adoption_origin']
    if run.get('execution_site')!='h1008' or origin.get('kind')!='h1008_official_zero_archive_v1':raise ValueError('REWARD_ZERO_ADOPTION_KIND')
    source_archive=Path(origin['archive']);source=source_archive.parent.parent.parent
    if digest(source_archive)!=origin['archive_sha256']:raise ValueError('REWARD_ZERO_SOURCE_ARCHIVE_CHANGED')
    source_run=json.loads((source/'logs/run-manifest.json').read_text());old=json.loads(source_archive.read_text())
    # Historical verifier may run after training; validate the recorded initial
    # binding without rewriting the actual current completed-update metadata.
    initial=dict(run,completed_updates=0);initial.pop('full_state_restore',None);validate_zero(initial,source_run,old)
    if live['round']!=0 or live['run_id']!=run['run_id'] or live['encoder']!=run['initial_model']:raise ValueError('REWARD_ZERO_TARGET_BINDING')
    if json.loads(archive_path.read_text())!=live:raise ValueError('REWARD_ZERO_TARGET_ARCHIVE_DIFFERS')
    for key in ('encoder_sha256','head_sha256','calibration_report','label_proof','dump_sha256','calibration_report_sha256'):
        if live[key]!=old[key]:raise ValueError('REWARD_ZERO_CALIBRATION_CHANGED:'+key)
    receipt=json.loads((base/'logs/initial-zero-adoption-receipt.json').read_text())
    if receipt['run_id']!=run['run_id'] or receipt['version']!=live['version']:raise ValueError('REWARD_ZERO_RECEIPT_BINDING')
    for name,sha in receipt['source_target_files_full_sha'].items():
        if digest(source_archive.parent/name)!=sha or digest(archive_path.parent/name)!=sha:raise ValueError('REWARD_ZERO_SOURCE_TARGET_FULL_SHA:'+name)
    return receipt
