import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from reproduction.hpc3_qprobe import storage_observer_sigma3_after_r2_permission_repaired as r


class RetryTests(unittest.TestCase):
    def run_worker(self,codes):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();own=dict(pid=os.getpid(),sid=os.getpid(),uid=r.UID,start_ticks=10)
            children=[]
            class Child:
                def __init__(self,index,rc,message):
                    self.pid=1000000+index;self.returncode=None;self.rc=rc;self.message=message;self.communicated=0
                def communicate(self,timeout):
                    self.communicated+=1;self.returncode=self.rc
                    return '123\t'+str(r.MEASURE)+'\n',self.message
                def poll(self):return self.returncode
                def kill(self):raise AssertionError('completed direct child must not be killed')
            def popen(*args,**kwargs):
                rc,message=codes[len(children)];child=Child(len(children),rc,message);children.append(child);return child
            def birth(pid):return own if pid==own['pid'] else dict(own,pid=pid,start_ticks=pid)
            claim=dict(source={},measure=str(r.MEASURE),uid=r.UID)
            with patch.object(r,'CONTROL',root),patch.object(r,'validate',return_value={}),\
                 patch.object(r,'owned'),patch.object(r,'read_json',return_value=(claim,'claimsha')),\
                 patch.object(r,'birth',side_effect=birth),patch.object(r.subprocess,'Popen',side_effect=popen),\
                 patch.object(Path,'read_text',return_value='du) S '+str(os.getpid())):
                result=r.worker()
            return result,children,[json.loads(p.read_text()) for p in sorted(root.glob('du-attempt-*-result.json'))]

    def test_enoent_then_clean_actual_measurement(self):
        message="du: cannot access '"+str(r.MEASURE)+"/logs/x.tmp': No such file or directory\n"
        value,children,attempts=self.run_worker([(1,message),(0,'')])
        self.assertEqual(value['status'],'ACTUAL_DU_COMPLETED0_USAGE_MEASURED')
        self.assertEqual(value['allocated_bytes'],123);self.assertEqual(len(attempts),2)
        self.assertEqual([x['actual_exit_code'] for x in attempts],[1,0])
        self.assertEqual([x.communicated for x in children],[1,1])
        self.assertEqual(value['du_process']['pid'],children[-1].pid)

    def test_permission_failure_is_never_retried_or_accepted(self):
        message="du: cannot read directory '"+str(r.MEASURE)+"/tmp/private': Permission denied\n"
        value,children,attempts=self.run_worker([(1,message),(0,'')])
        self.assertEqual(value['status'],'ACTUAL_DU_FAILED_NO_USAGE_ACCEPTED')
        self.assertNotIn('allocated_bytes',value);self.assertEqual(len(children),1)
        self.assertEqual(len(attempts),1)

    def test_three_enoent_attempts_are_bounded_and_all_retained(self):
        message="du: cannot access '"+str(r.MEASURE)+"/tmp/x': No such file or directory\n"
        value,children,attempts=self.run_worker([(1,message)]*3+[(0,'')])
        self.assertEqual(value['status'],'ACTUAL_DU_FAILED_NO_USAGE_ACCEPTED')
        self.assertEqual(len(children),3);self.assertEqual(len(attempts),3)
        self.assertNotIn('allocated_bytes',value)

    def test_outside_scope_and_mixed_error_are_not_retryable(self):
        self.assertFalse(r.retryable_enoent("du: cannot access '/etc/x': No such file or directory\n",1))
        self.assertFalse(r.retryable_enoent('',1))
        self.assertFalse(r.retryable_enoent("du: cannot access '"+str(r.MEASURE)+"/x': No such file or directory\nPermission denied\n",1))


if __name__=='__main__':unittest.main()
