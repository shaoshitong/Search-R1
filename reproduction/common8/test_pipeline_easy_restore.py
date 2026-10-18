"""CPU entry dispatch for the actual CP5 manifest; no CUDA acceptance claim."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from reproduction.common8 import run_pipeline as module


class EasyRestorePipelineEntryTests(unittest.TestCase):
    def restored(self):
        # completed_updates is the pristine allocation counter. Saved progress
        # belongs to full_state_restore until actual worker/driver reload.
        return dict(profile='common8', completed_updates=0, initial_probe_version='run:0:saved',
            source_sha256={}, easy2560_selection=True, execution_site='hpc3', execution_backend='slurm',
            training_world_size=4, data_selection_mode='mixed_first', question_probe=None,
            actor_logical_micro_batch_per_rank=4,
            full_state_restore=dict(kind='easy2560_native4', completed_outer=5,
                                    reward_probe_version='run:0:saved'))

    def test_real_shape_zero_allocation_counter_and_saved_five_probe_pass(self):
        run = self.restored(); before = copy.deepcopy(run)
        with patch('search_g1.easy2560_restore.source', side_effect=AssertionError('no duplicate source scan')):
            module.assert_initial_state(run)
        self.assertEqual(run, before)
        self.assertEqual(run['completed_updates'], 0)
        self.assertEqual(run['full_state_restore']['completed_outer'], 5)

    def test_fresh_entry_retains_zero_counter_and_no_probe_constraints(self):
        run = dict(profile='common8', completed_updates=0, initial_probe_version=None)
        module.assert_initial_state(run)
        for key, value in (('completed_updates',5), ('profile','a800'), ('initial_probe_version','unexpected')):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                module.assert_initial_state(dict(run, **{key:value}))

    def test_saved_version_exception_requires_exact_easy_native4_boundary(self):
        changes = [('completed_updates',5), ('completed_updates',200), ('easy2560_selection',False),
            ('training_world_size',8), ('execution_site','h1008'), ('actor_logical_micro_batch_per_rank',8),
            ('question_probe',{'enabled':True}), ('initial_probe_version',None),
            ('initial_probe_version','other'), ('restore_only_role','easy2560_final200_restore_only')]
        for key, value in changes:
            with self.subTest(key=key, value=value), self.assertRaises(AssertionError):
                module.assert_initial_state(dict(self.restored(), **{key:value}))
        for key, value in [('kind','other'), ('kind','easy2560_final200_restore_only'),
                           ('completed_outer',0), ('completed_outer',200), ('completed_outer',True),
                           ('reward_probe_version',''), ('reward_probe_version',None),
                           ('reward_probe_version','other')]:
            run = self.restored(); run['full_state_restore'][key] = value
            with self.subTest(binding=key,value=value), self.assertRaises(AssertionError):
                module.assert_initial_state(run)

    def test_every_retained_completed_outer_before200_can_enter_without_repeating_source_scan(self):
        for outer in (1,6,17,50,149,199):
            with self.subTest(outer=outer):
                run=self.restored();run['full_state_restore']['completed_outer']=outer
                with patch('search_g1.easy2560_restore.source',side_effect=AssertionError('do not repeat CPU readback')):
                    module.assert_initial_state(run)
                self.assertEqual(run['full_state_restore']['completed_outer'],outer)

    def test_actual_main_reaches_existing_allocation_and_pipeline_without_source_reload(self):
        for restored in (False, True):
            with self.subTest(restored=restored), tempfile.TemporaryDirectory() as td:
                base = Path(td).resolve(); (base/'logs').mkdir()
                run = self.restored() if restored else dict(profile='common8', completed_updates=0,
                    initial_probe_version=None, source_sha256={})
                (base/'logs/run-manifest.json').write_text(json.dumps(run))
                locks = []; real_open = Path.open
                def tracked_open(path, *args, **kwargs):
                    stream = real_open(path,*args,**kwargs)
                    if path.name=='gpu-job.lock': locks.append(stream)
                    return stream
                with patch.object(sys,'argv',['run_pipeline','--base',str(base),'--export-port','18099']), \
                     patch.object(module,'supported_execution_role',return_value='production'), \
                     patch.object(Path,'open',tracked_open), \
                     patch.object(module.signal,'signal'), patch.object(module.socket,'socket'), \
                     patch.object(module,'binding',return_value=(None,{'TRES':'memory fixture'},None)), \
                     patch.object(module,'memory_bytes',return_value=512*2**30), \
                     patch('reproduction.common8.native_allocation.admit') as allocation, \
                     patch('reproduction.common8.disk_admission.admit') as disk, \
                     patch('search_g1.easy2560_restore.source') as source, \
                     patch.object(module,'Pipeline') as pipeline:
                    module.main()
                for lock in locks: lock.close()
                allocation.assert_called_once_with(base); disk.assert_called_once_with(base)
                pipeline.assert_called_once_with(base,run); pipeline.return_value.execute.assert_called_once_with(18099)
                source.assert_not_called()

    def test_restoring_probe_dispatch_uses_saved_install_and_never_fresh_fit(self):
        pipeline = module.Pipeline.__new__(module.Pipeline)
        pipeline.base = Path('/CPU-fixture-not-remote'); pipeline.run = self.restored()
        pipeline.stage = Mock(); pipeline.start = Mock(side_effect=AssertionError('must not refit'))
        with patch('search_g1.easy2560_restore.install_restored_reward_probe',
                   return_value=dict(version='run:0:saved',round=0)) as install:
            pipeline.initial_probe_calibration()
        install.assert_called_once_with(pipeline.base,pipeline.run)
        pipeline.start.assert_not_called()
        self.assertEqual(pipeline.stage.call_args.args[0], 'RESTORING_SAVED_EASY_GROUNDING_PROBE_NO_NEW_CALIBRATION')


if __name__=='__main__': unittest.main()
