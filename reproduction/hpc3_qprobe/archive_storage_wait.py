"""One archive worker waits for exact permission repair, then remeasures fully.

Runs inside the original archive-storage lock. No payload is created here.
Every complete measurement has its own retained actual supervisor/DU wait
receipt. Permission never admits partial usage or starts another archive task.
"""
import hashlib
import json
import os
from pathlib import Path
import stat
import time

from reproduction.hpc3_qprobe import final_control as ctl
from reproduction.hpc3_qprobe import repair_actual_storage_denials as repair
from reproduction.hpc3_qprobe import storage_observer_sigma3_archive100_fourth as observer

PROTOCOL='actual-full-du-repair-v1'


def need(ok,message):ctl.need(ok,'ARCHIVE_STORAGE_WAIT: '+message)


def repair_evidence(failure_path,failure_sha,source_commit):
    """Real MOU result plus current JXU inode/access readback; no mutation."""
    value,roots=repair.failure(failure_path,failure_sha)
    directory=ctl.ROOT/'preparation'/('storage-actual-denial-repair-'+failure_sha[:24])
    if not (directory/'result.json').exists():return None
    result,result_sha=repair.document(directory/'result.json',21930)
    intent,intent_sha=repair.document(directory/'intent.json',21930)
    need(intent['source_commit']==result['source_commit']==source_commit and
         intent['failure_path']==str(failure_path) and intent['failure_sha256']==result['failure_sha256']==failure_sha and
         intent['roots']==[str(p) for p in roots] and intent['original_stderr_sha256']==value['stderr_sha256'] and
         result['status']=='ACTUAL_REPORTED_DENIED_SHARED_DIRECTORIES_EXACT_FD_REPAIRED' and
         result['recursive_chmod'] is False and result['regular_file_actions']==result['training_actions']==0 and
         repair.birth(intent['actual_process']['pid'])!=intent['actual_process'] and
         intent['created_unix']<=result['completed_unix'] and result['completed_unix']>=value['completed_unix'],
         'repair result source/scope/birth differs')
    repo,publication=ctl.source(source_commit)
    published=ctl.read(Path(publication['publication_path']))
    need(hashlib.sha256((repo/'reproduction/hpc3_qprobe/repair_actual_storage_denials.py').read_bytes()).hexdigest()==
         published['files']['reproduction/hpc3_qprobe/repair_actual_storage_denials.py'],'repair code publication differs')
    targets=intent['targets'];changed=result['changed'];missing=intent.get('missing_reported_roots',[])
    need(result.get('missing_reported_roots',[])==missing and len(missing)==len(set(missing)) and
         set(missing)<=set(intent['roots']) and len(changed)==len(targets) and
         len({x['path'] for x in targets})==len(targets),'repair member set differs')
    for index,(row,actual) in enumerate(zip(targets,changed),1):
        path=Path(row['path'])
        need(any(path==root or root in path.parents for root in roots) and
             actual==dict(path=row['path'],inode=row['inode'],before=0o700,after=0o750) and
             path.resolve()==path and repair.matches(path.lstat(),row,0o750),'current repaired inode/mode differs')
        retained,_=repair.document(directory/('changed-'+str(index)+'.json'),21930)
        need(retained==actual,'retained exact-FD mutation receipt differs')
    inspected=0
    for root in roots:
        if str(root) in missing:
            need(not root.exists() and not root.is_symlink(),'removed reported root reappeared; fresh diagnosis needed')
            continue
        todo=[root]
        while todo:
            path=todo.pop();s=path.lstat();inspected+=1
            need(inspected<=1024 and path.resolve()==path and stat.S_ISDIR(s.st_mode) and
                 s.st_uid==21930 and s.st_gid==21919 and os.access(path,os.R_OK|os.X_OK),
                 'reported root is not traversable by JXU')
            for entry in os.scandir(path):
                p=Path(entry.path);s=p.lstat()
                need(not stat.S_ISLNK(s.st_mode),'reported tree acquired a symlink')
                if stat.S_ISDIR(s.st_mode):todo.append(p)
                else:need(stat.S_ISREG(s.st_mode),'reported tree member type differs')
    return dict(path=str(directory/'result.json'),sha256=result_sha,intent_sha256=intent_sha,
                same_inode_readbacks=len(changed),traversable_directories=inspected,checked_unix=time.time())


def accepted_usage(actual):
    value=actual.get('result') or {}
    need(actual['status']=='ACTUAL_DU_COMPLETED0_USAGE_MEASURED' and not actual['actual_live_birth'] and
         not actual['actual_du_live_birth'] and value['actual_du_exit_code']==0 and value['stderr']=='' and
         value['stdout_truncated'] is False and value['stderr_truncated'] is False and
         type(value['allocated_bytes']) is int,'complete actual held0 DU required')
    return value['allocated_bytes']


def measure(claim_path,token,run_id,step):
    claim_path=Path(claim_path);folder=claim_path.parent
    claim=ctl.read(claim_path);claim_sha=ctl.digest(claim_path)
    need(step in (100,150,200) and claim['token']==token and claim['step']==step and
         claim['producer']['run_id']==run_id and claim.get('storage_wait_protocol')==PROTOCOL,
         'actual archive claim scope differs')
    own=ctl.birth(ctl.process(os.getpid()));task=ctl.read(folder/'task-binding.json')
    need(task['actual_process']==own and task['claim_sha256']==claim_sha,'actual held archive child differs')
    payload=ctl.ROOT/'archives'/run_id/('global_step_'+str(step))
    first={100:1,150:17,200:33}[step]
    for index,slot in enumerate(range(first,first+16),1):
        need(not payload.exists() and not payload.is_symlink(),'payload appeared before storage admission')
        core=observer.load_core(slot);binding_path=folder/('storage-measurement-'+str(index)+'-binding.json')
        need(not core.CONTROL.exists() and not core.CONTROL.is_symlink() and not binding_path.exists(),
             'existing measurement scope is inspect-only; never replay')
        ctl.publish(binding_path,dict(claim_sha256=claim_sha,token=token,archive_child=own,
            control=str(core.CONTROL),slot=slot,source_commit=claim['source']['commit'],started_unix=time.time()))
        launched,child=observer.start_held(core)
        observed_birth=launched.get('process')
        need(observed_birth is not None and observed_birth['pid']==child.pid and
             observed_birth['uid']==ctl.OWNER,'held observer child birth differs')
        while True:
            actual=core.inspect();value=actual.get('result')
            need(actual.get('claim',{}).get('parent_process')==own,'measurement parent birth differs')
            if value is not None and not actual['actual_du_live_birth']:
                # A detached observer is still our direct child. Reap it before
                # using birth predicates, which deliberately include zombies.
                code=child.wait(timeout=30)
                ctl.publish(folder/('storage-measurement-'+str(index)+'-held.json'),dict(
                    claim_sha256=claim_sha,token=token,archive_child=own,observer_birth=observed_birth,
                    actual_observer_wait_exit_code=code,finished_unix=time.time()))
                need(code==0,'held observer exited abnormally')
                actual=core.inspect();value=actual['result']
                need(not actual['actual_live_birth'] and not actual['actual_du_live_birth'],
                     'observer/DU still live after held wait')
                break
            need(actual['actual_live_birth'] or actual['actual_du_live_birth'],
                 'measurement outcome unknown; retain claim and refuse replay')
            need(child.poll() is None,'held observer died without a complete DU result; no replay')
            time.sleep(5)
        failure_path=core.CONTROL/'result.json';failure_sha=ctl.digest(failure_path)
        if actual['status']=='ACTUAL_DU_COMPLETED0_USAGE_MEASURED':
            allocated=accepted_usage(actual)
            ctl.publish(folder/('storage-measurement-'+str(index)+'-accepted.json'),dict(
                claim_sha256=claim_sha,token=token,allocated_bytes=allocated,result_path=str(failure_path),
                result_sha256=failure_sha,actual_process=own,accepted_unix=time.time()))
            return allocated
        # This parser accepts only actual in-scope reported permission roots,
        # optionally accompanied by ordinary disappearing temporary entries.
        repair.failure(failure_path,failure_sha)
        ctl.publish(folder/('storage-measurement-'+str(index)+'-waiting.json'),dict(
            phase='WAITING_STORAGE_REPAIR',claim_sha256=claim_sha,token=token,actual_process=own,
            failure_path=str(failure_path),failure_sha256=failure_sha,payload_created=False,checked_unix=time.time()))
        while True:
            need(ctl.digest(claim_path)==claim_sha and ctl.digest(failure_path)==failure_sha and
                 not payload.exists() and not payload.is_symlink(),'waiting claim/failure/payload changed')
            repaired=repair_evidence(failure_path,failure_sha,claim['source']['commit'])
            if repaired is not None:break
            time.sleep(10)
        ctl.publish(folder/('storage-measurement-'+str(index)+'-repair-accepted.json'),repaired)
    raise ValueError('Sixteen actual permission failures; retain the live diagnosis, no archive acceptance')
