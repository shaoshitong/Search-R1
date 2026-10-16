"""Published CPU test gates: isolated regressions and real shared-FS storage.

Historical recovery fixtures deliberately remap the actual experiment root.
Their scratch directories must therefore remain outside that root. Only the
six real storage/child/lock tests run under the actual shared experiment FS.
No producer, checkpoint, RAW, archive claim or training job is modified.
"""
import argparse
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest

from reproduction.hpc3_qprobe import owner_partition_storage_v2 as storage

MODULES=('test_owner_partition_storage','test_recover_sigma3_archive150_first',
    'test_recover_sigma3_archive100_fourth','test_recover_sigma3_archive100_third',
    'test_recover_sigma3_archive100_second','test_recover_sigma3_archive100',
    'test_recover_sigma3_archive50_second','test_recover_sigma3_archive50','test_recover_sigma3_archive5',
    'test_recovery_lineage_stage','test_archive_storage_wait','test_followthrough_sigma3',
    'test_sigma3_archive_binding','test_sigma3_archive100_route','test_independent_goal_control_sigma3',
    'test_independent_goal_acceptance_sigma3','test_sigma3_archive150_route',
    'test_archive150_visibility_failure','test_owner_partition_storage_v2',
    'test_recover_sigma3_archive150_second','test_local_owner_partition_driver','test_sigma3_archive150_second_route')
RELATIVE='reproduction/hpc3_qprobe/protocol_tests_v2.py'


def guard(commit,owner):
    repo,source=storage.code_guard(commit,owner)
    storage.need(Path(__file__).resolve()==repo/RELATIVE,'test entry is not its published immutable source')
    return source


def run(commit,shared=False):
    source=guard(commit,storage.JXU)
    if shared:
        folder=storage.ROOT/'observations'/('protocol-v2-shared-tests-'+commit)
        folder.mkdir(mode=0o770,exist_ok=True);tempfile.tempdir=str(folder)
        names=['reproduction.hpc3_qprobe.test_owner_partition_storage_v2.'+name
               for name in ('ActualGNU','ActualChildren','ActualLaunchAndLock')]
    else:
        tempfile.tempdir='/tmp';folder=Path(tempfile.gettempdir()).resolve()
        storage.need(storage.ROOT!=folder and storage.ROOT not in folder.parents,
                     'historical regression fixtures must be outside the production namespace')
        names=['reproduction.hpc3_qprobe.'+name for name in MODULES]
    suite=unittest.defaultTestLoader.loadTestsFromNames(names)
    output=io.StringIO();result=unittest.TextTestRunner(stream=output,verbosity=2).run(suite)
    ok=result.wasSuccessful() and not result.skipped
    return dict(status='ACTUAL_PROTOCOL_TEST_GATE_PASS' if ok else 'ACTUAL_PROTOCOL_TEST_GATE_FAILED',
        scope='actual_shared_filesystem_storage' if shared else 'isolated_full_regression',source=source,
        tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),skipped=result.skipped,
        failed_names=[str(test) for test,_ in result.failures+result.errors],output=output.getvalue(),
        tempfile_root=str(folder),filesystem_device=folder.stat().st_dev,checked_unix=time.time(),
        actor_model_loaded=False,training_actions=0,permission_changes=0)


def cross_owner_lock(commit,owner):
    source=guard(commit,owner)
    folder=storage.ROOT/'observations'/('protocol-v2-cross-owner-lock-'+commit)
    path=folder/'owner-admission.lock'
    if owner==storage.JXU:
        storage.need(not folder.exists() and not folder.is_symlink(),'existing cross-owner probe is inspect-only')
        folder.mkdir(mode=0o770)
        storage.publish(path,dict(role='protocol_v2_shared_readonly_fd_probe',source=source,created_unix=time.time()))
    value,proof=storage.document(path,(storage.JXU,),protected=True)
    storage.need(value['source']==source and value['role']=='protocol_v2_shared_readonly_fd_probe','cross-owner probe source differs')
    with storage.admission_lock(folder/'request.json'):
        return dict(status='ACTUAL_0640_SHARED_READONLY_EXCLUSIVE_FLOCK_ACQUIRED',source=source,lock=proof,
            actual_process=storage.birth(storage.process(os.getpid())),host=storage.host_binding(),
            filesystem_device=path.stat().st_dev,mode=path.stat().st_mode&0o777,checked_unix=time.time(),
            training_actions=0,permission_changes=0)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('regression','shared-storage','jxu-lock-probe','mou-lock-probe'))
    parser.add_argument('--commit',required=True);args=parser.parse_args()
    if args.mode in ('regression','shared-storage'):
        value=run(args.commit,shared=args.mode=='shared-storage');code=0 if value['status']=='ACTUAL_PROTOCOL_TEST_GATE_PASS' else 1
    else:
        value=cross_owner_lock(args.commit,storage.JXU if args.mode=='jxu-lock-probe' else storage.MOU);code=0
    print(json.dumps(value),flush=True);sys.exit(code)
