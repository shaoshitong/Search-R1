"""Actual local observer processes exercise the archive wait/reap transition."""
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from reproduction.hpc3_qprobe import archive_storage_wait as wait
from reproduction.hpc3_qprobe import repair_actual_storage_denials as repair


class StorageWait(unittest.TestCase):
    def test_partial_usage_never_accepts(self):
        for status,exit_code,stderr in [('ACTUAL_DU_FAILED_NO_USAGE_ACCEPTED',1,'Permission denied'),
                                       ('ACTUAL_DU_COMPLETED0_USAGE_MEASURED',1,''),
                                       ('ACTUAL_DU_COMPLETED0_USAGE_MEASURED',0,'Permission denied')]:
            value=dict(actual_du_exit_code=exit_code,stderr=stderr,allocated_bytes=1,
                       stdout_truncated=False,stderr_truncated=False)
            with self.subTest(status=status,exit_code=exit_code,stderr=stderr),self.assertRaises(ValueError):
                wait.accepted_usage(dict(status=status,result=value,actual_live_birth=False,actual_du_live_birth=False))

    def run_actual_observers(self,permission_first,inspect_fail=False):
        with tempfile.TemporaryDirectory() as temporary,ExitStack() as stack:
            root=Path(temporary);folder=root/'final-control/run/archive-100';folder.mkdir(parents=True)
            own=dict(pid=os.getpid(),start_ticks=123,sid=os.getsid(0),uid=os.getuid())
            claim=dict(token='token',step=100,producer={'run_id':'run'},storage_wait_protocol=wait.PROTOCOL,
                       source={'commit':'c'*40})
            claim_path=folder/'submit-claim.json';claim_path.write_text(json.dumps(claim))
            digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
            (folder/'task-binding.json').write_text(json.dumps(dict(actual_process=own,claim_sha256=digest(claim_path))))
            payload=root/'archives/run/global_step_100';children=[];cores=[];repair_calls=[]
            def core(slot):
                item=SimpleNamespace(CONTROL=root/('observer-'+str(slot)),subprocess=subprocess)
                state={'pid':None,'inspections':0}
                denied=permission_first and not cores
                value=dict(status='ACTUAL_DU_FAILED_NO_USAGE_ACCEPTED' if denied else 'ACTUAL_DU_COMPLETED0_USAGE_MEASURED',
                           actual_du_exit_code=1 if denied else 0,stderr='Permission denied' if denied else '',
                           allocated_bytes=1540000000000,stdout_truncated=False,stderr_truncated=False)
                def inspect():
                    state['inspections']+=1
                    if inspect_fail and state['inspections']>1:raise OSError('actual inspect failure after Popen')
                    path=item.CONTROL/'result.json';result=json.loads(path.read_text()) if path.exists() else None
                    try:os.kill(state['pid'],0);live=True
                    except ProcessLookupError:live=False
                    return dict(status=result['status'] if result else 'ACTUAL_OBSERVER_LIVE',
                        result=result,actual_live_birth=live,actual_du_live_birth=False,
                        claim={'parent_process':own},process=dict(own,pid=state['pid']))
                def start():
                    item.CONTROL.mkdir()
                    code=('import pathlib;target=pathlib.Path('+repr(str(item.CONTROL/'result.json'))+');'
                          'temporary=target.with_suffix(".tmp");temporary.write_text('+repr(json.dumps(value))+');'
                          'temporary.replace(target)')
                    child=item.subprocess.Popen([sys.executable,'-c',code]);children.append(child);state['pid']=child.pid
                    return inspect()
                item.inspect=inspect;item.start=start;cores.append(item);return item
            def repaired(*args):
                self.assertFalse(payload.exists());self.assertEqual(len(cores),1)
                self.assertEqual(children[0].returncode,0)
                repair_calls.append(args)
                return None if len(repair_calls)==1 else {'same_inode_readbacks':1,'sha256':'a'*64}
            for target,name,value in [(wait.ctl,'ROOT',root),(wait.ctl,'OWNER',os.getuid()),
                (wait.ctl,'read',lambda p:json.loads(Path(p).read_text())),(wait.ctl,'digest',digest),
                (wait.ctl,'process',lambda pid:own),(wait.ctl,'birth',lambda p:p),
                (wait.observer,'load_core',core),(wait.repair,'failure',lambda *args:({},[])),
                (wait,'repair_evidence',repaired),(wait.time,'sleep',lambda _:time.perf_counter())]:
                stack.enter_context(patch.object(target,name,value))
            if inspect_fail:
                with self.assertRaisesRegex(OSError,'inspect failure'):wait.measure(claim_path,'token','run',100)
                record=json.loads((folder/'storage-measurement-1-exception.json').read_text())
                self.assertFalse(record['cleanup_pending'])
                self.assertEqual(record['observer_cleanup']['actual_observer_wait_exit_code'],0)
                self.assertEqual(children[0].returncode,0)
                self.assertFalse(payload.exists())
                return
            self.assertEqual(wait.measure(claim_path,'token','run',100),1540000000000)
            self.assertFalse(payload.exists())
            self.assertEqual(len(children),2 if permission_first else 1)
            self.assertTrue(all(child.returncode==0 for child in children))
            for index in range(1,len(children)+1):
                held=json.loads((folder/('storage-measurement-'+str(index)+'-held.json')).read_text())
                self.assertEqual(held['actual_observer_wait_exit_code'],0)
                self.assertEqual(held['archive_child'],own)
            self.assertEqual(len(repair_calls),2 if permission_first else 0)

    def test_completed_observer_is_reaped_before_acceptance_with_live_archive_parent(self):
        self.run_actual_observers(False)

    def test_permission_wait_creates_no_payload_and_same_parent_remeasures_only_after_repair(self):
        self.run_actual_observers(True)

    def test_actual_inspect_failure_waits_for_the_held_child_before_unwinding(self):
        self.run_actual_observers(False,inspect_fail=True)

    def test_removed_reported_temporary_root_is_recorded_without_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'gone'
            existing,missing=repair.existing_reported_roots([path])
            self.assertEqual(existing,[]);self.assertEqual(missing,[str(path)])
            path.symlink_to(Path(temporary)/'absent')
            with self.assertRaises(ValueError):repair.existing_reported_roots([path])


if __name__=='__main__':unittest.main()
