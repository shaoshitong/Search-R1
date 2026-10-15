"""Local protocol checks for isolated observation slots; no remote work."""
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import unittest

from reproduction.hpc3_qprobe import storage_observer_sigma3_archive100_fourth as wrapper
from reproduction.hpc3_qprobe import storage_observer_sigma3_after_actual_denial_repair as canonical


class ObserverNamespace(unittest.TestCase):
    def test_actual_child_is_retained_and_reaped_while_parent_remains_live(self):
        core=SimpleNamespace(subprocess=subprocess)
        def start():
            child=core.subprocess.Popen([sys.executable,'-c','print("actual-observer-complete")'],stdout=subprocess.PIPE)
            return {'pid':child.pid}
        core.start=start
        value,held=wrapper.start_held(core)
        self.assertEqual(value['pid'],held.pid)
        self.assertIs(core.subprocess,subprocess)
        output,_=held.communicate(timeout=10)
        self.assertEqual(held.wait(),0)
        self.assertEqual(output,b'actual-observer-complete\n')
        # A second wait uses the retained status; no child is relaunched.
        self.assertEqual(held.wait(),0)

    def test_private_configuration_does_not_patch_the_canonical_worker(self):
        before=(canonical.__file__,canonical.CONTROL,canonical.SLOT,canonical.RELATIVE_SOURCE)
        private=SimpleNamespace();entry=Path('/fixture/source/reproduction/hpc3_qprobe/entry.py')
        result=wrapper.configure(private,17,entry)
        self.assertIs(result,private)
        self.assertEqual(private.CONTROL,wrapper.ROOT/'observations'/(wrapper.PREFIX+'source-17'))
        self.assertEqual(private.RELATIVE_SOURCE,wrapper.RELATIVE)
        self.assertEqual(private.__file__,str(entry))
        self.assertEqual((canonical.__file__,canonical.CONTROL,canonical.SLOT,canonical.RELATIVE_SOURCE),before)

    def test_ambiguous_slots_refuse(self):
        for slot in (0,65,True,'1',1.0):
            with self.subTest(slot=slot),self.assertRaises(ValueError):wrapper.configure(SimpleNamespace(),slot,Path('/fixture'))

    def test_same_canonical_full_output_and_retry_rules(self):
        stdout='1540000000000\t'+str(canonical.MEASURE)+'\n'
        self.assertEqual(canonical.allocated_usage(stdout,'',0),1540000000000)
        denied="du: cannot read directory '/data/user/jxu957/experiments/tmp': Permission denied\n"
        missing="du: cannot access '/data/user/jxu957/experiments/tmp': No such file or directory\n"
        self.assertIsNone(canonical.allocated_usage(stdout,denied,1))
        self.assertFalse(canonical.retryable_enoent(denied,1))
        self.assertFalse(canonical.retryable_enoent(denied+missing,1))
        self.assertTrue(canonical.retryable_enoent(missing,1))


if __name__=='__main__':unittest.main()
