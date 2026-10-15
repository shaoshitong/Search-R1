import unittest
from types import SimpleNamespace
from unittest.mock import patch

from reproduction.hpc3_qprobe import independent_goal_acceptance_sigma3 as reader
from reproduction.hpc3_qprobe import independent_goal_control_sigma3 as holder


class SecondRecoveryRoute(unittest.TestCase):
    def test_explicit_second_source_uses_distinct_module_directory_and_scope(self):
        second=reader.ARCHIVE50_SECOND_RECOVERY_SOURCE
        args=(second,reader.ARCHIVE5_RECOVERY_SOURCE,'f979d53cdb0d348e87c5877d84403252e85cd784',second)
        name,folder=reader.followthrough_route(*args)
        self.assertEqual(name,'recover_sigma3_archive50_second')
        self.assertEqual(folder.name,'721485-archive50-recovery-second-20261015')
        self.assertTrue(holder.scope(*args[1:]).name.endswith('-archive50-recovery-second'))
        first=reader.ARCHIVE50_FIRST_RECOVERY_SOURCE
        first_args=(first,*args[1:3],first)
        self.assertEqual(reader.followthrough_route(*first_args)[0],'recover_sigma3_archive50')
        self.assertNotEqual(holder.scope(*args[1:]),holder.scope(*first_args[1:]))
        with self.assertRaises(ValueError): reader.followthrough_route(second,args[1],second,second)

    def test_prior_chain_is_reconsumed_and_not_only_trusted_as_embedded_json(self):
        prior={'source':{'commit':reader.ARCHIVE50_FIRST_RECOVERY_SOURCE},'preserved':True}
        recovery={'prior_archive50_recovery':prior}
        old=SimpleNamespace(consume=lambda commit:prior)
        with patch.object(reader,'load_fixed',return_value=(old,{})) as load:
            self.assertEqual(reader.prior_archive50_evidence(recovery,reader.ARCHIVE50_SECOND_RECOVERY_SOURCE),prior)
            load.assert_called_once_with(reader.ARCHIVE50_FIRST_RECOVERY_SOURCE,'recover_sigma3_archive50','control')
        with patch.object(reader,'load_fixed',return_value=(SimpleNamespace(consume=lambda commit:{}),{})):
            with self.assertRaises(ValueError):reader.prior_archive50_evidence(recovery,reader.ARCHIVE50_SECOND_RECOVERY_SOURCE)
        with self.assertRaises(ValueError):reader.prior_archive50_evidence(recovery,reader.ARCHIVE50_FIRST_RECOVERY_SOURCE)


if __name__=='__main__':unittest.main()
