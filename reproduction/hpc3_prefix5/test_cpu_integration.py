"""CPU-only publication and old-kernel lifecycle checks; no GPU allocation."""
import errno
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from reproduction.prefix_selection200 import i1_i6_joint_resume as recovery
from reproduction.prefix_only_correctness_ab.launch import torchrun_owned_class
from reproduction.a800 import owned_processes


class HPC3CPUIntegrationTests(unittest.TestCase):
    def test_only_own_immutable_publication_alias_is_admitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp).resolve();source=base/'source'/('a'*40);source.mkdir(parents=True)
            (source/'entry.py').write_text('meaning=42\n');alias=base/'repo';alias.symlink_to(source)
            with patch.object(recovery,'HPC3_BASE',base):
                self.assertEqual(recovery._code_inventory(alias),recovery._code_inventory(source))
                self.assertEqual(recovery.source_manifest(alias,'a'*40)['source_commit'],'a'*40)
                alias.unlink();foreign=base/'foreign';foreign.mkdir();alias.symlink_to(foreign)
                with self.assertRaisesRegex(ValueError,'immutable publication'):recovery._code_inventory(alias)
                alias.unlink();alias.symlink_to(source)
                other=base/'other';other.symlink_to(source)
                with self.assertRaisesRegex(ValueError,'canonical'):recovery._code_inventory(other)
                # This exception applies to source inventory only, never a
                # checkpoint or optimizer payload path.
                with self.assertRaisesRegex(ValueError,'canonical'):recovery.canonical_path(alias)

    @unittest.skipUnless(sys.platform=='linux','Actual /proc ownership requires Linux')
    def test_no_pidfd_leader_gets_grace_to_reap_own_descendant(self):
        script='''import pathlib,signal,subprocess,sys,time
child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)'],start_new_session=True)
def finish(sig,frame):
 child.terminate();child.wait(timeout=5);raise SystemExit(0)
signal.signal(signal.SIGINT,finish);signal.signal(signal.SIGTERM,finish)
pathlib.Path(sys.argv[1]).write_text(str(child.pid))
while True:time.sleep(.1)
'''
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ready=root/'ready';owned=torchrun_owned_class()(root/'ledger.json','cpu-integration')
            p=owned.start('leader',[sys.executable,'-c',script,str(ready)],os.environ.copy(),str(root/'leader.log'),str(root))
            try:
                until=time.monotonic()+10
                while not ready.exists() and p.poll() is None and time.monotonic()<until:time.sleep(.02)
                self.assertTrue(ready.exists());child=int(ready.read_text())
                self.assertIn(child,{r['pid'] for r in owned.observe('leader')})
                with patch.object(owned_processes,'_pidfd_open',side_effect=OSError(errno.ENOSYS,'CPU fixture old kernel')), \
                     patch.object(owned_processes,'_signal_direct_child',wraps=owned_processes._signal_direct_child) as direct:
                    owned.stop('leader')
                self.assertEqual(owned.jobs['leader']['state'],'STOPPED')
                self.assertTrue(direct.called)
                self.assertTrue(all(c.args[0]['pid']==p.pid for c in direct.call_args_list))
                self.assertTrue(owned.jobs['leader']['descendant_signal_deferred'])
                self.assertEqual(owned.observe('leader'),[])
            finally:
                if p.poll() is None:
                    owned_processes.signal_owned(owned.jobs['leader']['leader'],signal.SIGTERM,process=p)
                    p.wait(timeout=10)


if __name__=='__main__':unittest.main()
