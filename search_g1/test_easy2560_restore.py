"""Local CPU file/provenance mechanics with injected Torch load/rank validation.

No CUDA/Adam tensor acceptance is claimed by these synthetic byte fixtures;
production create_receipt uses the unchanged qprobe validate_rank_state on HPC.
"""
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from search_g1 import easy2560_restore as module


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + '\n')


class Fixture:
    def __init__(self, root):
        root = Path(root).resolve()
        self.base = root/'source'; self.cp = self.base/'checkpoints/search-g1-common8/actor/global_step_5'
        self.state = self.cp/'training_state'; self.state.mkdir(parents=True)
        self.output = Path(root)/'receipt.json'; self.run_id = 'd'*32
        self.eligible = self.base/'assets/eligible.json'; write(self.eligible, dict(records=list(range(8))))
        self.encoder = self.base/'official'; self.encoder.mkdir(); (self.encoder/'model.bin').write_bytes(b'fixture official weights')
        self.run = dict(profile='common8', easy2560_selection=True, execution_site='hpc3', execution_backend='slurm',
            training_world_size=4, actor_micro_batch_per_rank=16, actor_logical_micro_batch_per_rank=4,
            data_selection_mode='mixed_first', question_probe=None, all_correct_virtual_zero=False,
            selection_candidate_questions=2560, selection_selected_questions=512, selection_fixed_candidate_budget=True,
            selection_eligible_pool_path=str(self.eligible), run_id=self.run_id, source_commit='8'*40,
            completed_updates=0, input_sha256={'train.parquet':'a'*64}, initial_model_sha256={'model.bin':module.digest(self.encoder/'model.bin')},
            model_revision='immutable-model', budget=dict(world_size=4, learning_rate=1e-6),
            retriever_topk=3, retriever_version='immutable-index', retriever_manifest_sha256='b'*64)
        write(self.base/'logs/run-manifest.json', self.run)
        write(self.base/'logs/native-allocation-admission.json', dict(accepted=True, run_id=self.run_id,
                                                                  training_world_size=4, slurm_job_id='730226'))
        self.ids = [f'{n:032x}' for n in range(1, 6)]
        self.rank_states = []
        for rank in range(4):
            saved = dict(rank=rank, world_size=4, run_id=self.run_id, source_commit=self.run['source_commit'],
                         actual_optimizer_writes=100, completed_outer=5, last_attempt_index=5,
                         used_attempt_ids=self.ids, policy_rejections=0, rng={'fixture_cpu_rng': [rank]},
                         scheduler=dict(last_epoch=5, _step_count=6), optimizer={'fixture_cpu_only': True})
            self.rank_states.append(saved); write(self.state/f'rank-{rank}.pt', saved)
        self.progress = dict(completed_outer=5, cumulative_writes=100, attempt_index=5,
                             aborted_attempts=0, full_budget_outer=5)
        self.version = self.run_id+':0:original'
        self.driver = dict(schema_version=1, easy2560_selection=True, selection_mode='mixed_first',
            question_probe=None, run_id=self.run_id, source_commit=self.run['source_commit'], completed_outer=5,
            attempt_progress=self.progress, probe_version=self.version, probe_round=0, reward_refresh_pending=False,
            rng=dict(torch_cpu=[1], numpy=[2], python=[3]),
            candidate_stream=dict(candidate_order=list(range(8)), candidate_cursor=4, epoch=1600,
                consumed_source_rows=12800, pending_source=None, pending_offset=0,
                prefix20_pool_binding=dict(eligible_questions=8, eligible_pool_sha256=module.digest(self.eligible),
                                          canonical_train_sha256='a'*64)))
        write(self.state/'driver.pt', self.driver)
        (self.cp/'weights.safetensors').write_bytes(b'synthetic HF bytes, not actual model tensors')
        write(self.cp/'config.json', {})
        write(self.cp/'model.safetensors.index.json', {'weight_map':{'fixture.weight':'weights.safetensors'}})
        self.refresh_checkpoint()
        folder = self.base/'artifacts/round_0-attempt-original'; heads = folder/'heads'; heads.mkdir(parents=True)
        for name in ('commit_probe.pkl', 'boundary_probe.pkl'):
            (heads/name).write_bytes(b'original fixture head '+name.encode())
        write(heads/'calibration-report.json', dict(original_calibration=True))
        probe = dict(round=0, run_id=self.run_id, version=self.version, encoder=str(self.encoder),
            commit=str(heads/'commit_probe.pkl'), boundary=str(heads/'boundary_probe.pkl'),
            head_sha256={n:module.digest(heads/n) for n in ('commit_probe.pkl','boundary_probe.pkl')},
            calibration_report_sha256=module.digest(heads/'calibration-report.json'),
            encoder_sha256={'model.bin':module.digest(self.encoder/'model.bin')})
        write(folder/'version-manifest.json', probe)
        updates = []; attempts = []
        for step, aid in enumerate(self.ids, 1):
            gate = self.base/'logs'/('behavior-probability-attempt-'+aid+'.json')
            write(gate, dict(run_id=self.run_id, attempt_id=aid, logical_update=step))
            write(self.base/'artifacts/mixed-selection'/(aid+'.json'), dict(step=step, candidate_fill_seconds=10))
            updates.append(dict(step=step, run_id=self.run_id, attempt_id=aid, attempt_index=step,
                optimizer_writes_before=(step-1)*20, optimizer_writes_after=step*20,
                actual_optimizer_writes=20, behavior_probability_gate_sha256=module.digest(gate)))
            attempts.append(dict(attempt_index=step, attempt_id=aid, run_id=self.run_id, slurm_job_id='730226'))
        self.lines('logs/actual-updates.jsonl', updates); self.lines('logs/actual-attempts.jsonl', attempts)
        for role in ('optimizer','scheduler'):
            for rank in range(4):
                self.lines(f'logs/{role}-rank-{rank}.jsonl', [dict(rank=rank, run_id=self.run_id,
                    context=dict(attempt_index=n)) for n in range(1, 6)])
        self.lines('logs/validation-identities.jsonl', [dict(step=5, split='stage'), dict(step=5, split='validation')])
        write(self.base/'logs/formal-inline-development-after.json', dict(run_id=self.run_id, step=5, weighted_em=142/512))

    def lines(self, relative, rows):
        p = self.base/relative; p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(''.join(json.dumps(r)+'\n' for r in rows))

    def refresh_checkpoint(self):
        rows = [dict(path=str(self.state/f'rank-{rank}.pt'), rank=rank, actual_optimizer_writes=100,run_id=self.run_id,
                     sha256=module.digest(self.state/f'rank-{rank}.pt')) for rank in range(4)]
        write(self.state/'manifest.json', dict(completed_outer=5, question_probe=None, worker_files=rows,
            driver_file=dict(path=str(self.state/'driver.pt'), sha256=module.digest(self.state/'driver.pt'))))
        write(self.cp/'complete.json', dict(step=5, run_id=self.run_id, checkpoint=str(self.cp),
            finite_all_saved_tensors=True, validated_tensor_count=1, actual_optimizer_writes_per_rank=100,
            files_sha256={str(p.relative_to(self.cp)):module.digest(p) for p in self.cp.rglob('*')
                          if p.is_file() and p.name!='complete.json'}))

    @staticmethod
    def rank_check(saved, rank, outer, writes):
        if (saved['actual_optimizer_writes'], saved['rank'], saved['world_size'], saved['completed_outer']) != (writes,rank,4,outer):
            raise ValueError('INJECTED_COUNTER_READBACK_MISMATCH')
        return dict(rank=rank, actual_optimizer_writes=writes, last_attempt_index=saved['last_attempt_index'],
                    used_attempt_ids=saved['used_attempt_ids'], policy_rejections=saved['policy_rejections'],
                    completed_outer=outer, scheduler_last_epoch=outer, scheduler_step_count=outer+1,
                    optimizer_tensors_finite=True)

    def create(self):
        with patch.object(module, '_load_cpu', side_effect=lambda path: json.loads(Path(path).read_text())), \
             patch.object(module, '_validate_rank', side_effect=self.rank_check):
            return module.create_receipt(self.base, self.cp, self.output)


class EasyRestoreTests(unittest.TestCase):
    def test_login_node_mount_device_is_not_a_cross_node_file_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            f=Fixture(folder);f.create();receipt=module._json(f.output)
            def other_mount(value):
                if isinstance(value,dict):
                    if 'device' in value:value['device']+=1
                    for item in value.values():other_mount(item)
                elif isinstance(value,list):
                    for item in value:other_mount(item)
            other_mount(receipt);write(f.output,receipt)
            # The unchanged shared files still have the pinned inode/owner,
            # size/mtime and SHA; a different mount number must not block them.
            module.source(f.output)
            record=receipt['files'][0]
            for field in ('inode','uid'):
                altered=dict(record);altered[field]+=1
                with self.assertRaisesRegex(ValueError,'SOURCE_FILE_IDENTITY_CHANGED'):
                    module._identity(altered)
            qa=f.base/'logs/formal-inline-development-after.json'
            raw=qa.read_bytes();before=qa.stat()
            qa.write_bytes(raw.replace(b'142',b'143') if b'142' in raw else raw[:-2]+b' \n')
            os.utime(qa,ns=(before.st_atime_ns,before.st_mtime_ns))
            with self.assertRaisesRegex(ValueError,'SOURCE_SMALL_SHA_CHANGED'):
                module._identity(receipt['historical_artifacts']['development']['after5'],rehash=True)

    def test_full_file_SHA_native4_boundary_history_and_no_invented_before(self):
        with tempfile.TemporaryDirectory() as folder:
            f=Fixture(folder); proof=f.create(); _,r,cp,base,parent=module.source(f.output)
            self.assertEqual(proof['status'],module.STATUS); self.assertEqual(r['world_size'],4)
            self.assertEqual(r['driver_state']['source_consumed'],12800)
            self.assertEqual(len(r['files']),len([p for p in cp.rglob('*') if p.is_file()]))
            self.assertTrue(r['checkpoint_saved_before_development']); self.assertFalse(r['post_callback_rng_captured'])
            self.assertFalse(r['actual_cuda_restore_accepted']); self.assertFalse(r['source_archive_receipt_available'])
            self.assertEqual(set(r['historical_artifacts']['development']),{'after5'})
            self.assertEqual(parent,f.run); self.assertEqual(base,f.base)
            required={'logs/actual-updates.jsonl','logs/actual-attempts.jsonl','logs/validation-identities.jsonl',
                      'checkpoint/complete.json','checkpoint/training_state/manifest.json'}
            self.assertTrue(required<=set(r['historical_files']))
            self.assertFalse(any('mixed-candidate-pools' in n for n in r['historical_files']))

    def test_changed_checkpoint_bytes_fail_CPU_publication(self):
        with tempfile.TemporaryDirectory() as folder:
            f=Fixture(folder); (f.cp/'weights.safetensors').write_bytes(b'changed HF bytes')
            with self.assertRaisesRegex(ValueError,'COMPLETE_SHA'): f.create()
            self.assertFalse(f.output.exists())

    def test_rank_mismatch_and_driver_cursor_not_silently_accepted(self):
        for mutation in ('rank','cursor'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as folder:
                f=Fixture(folder)
                if mutation=='rank':
                    row=f.rank_states[3]; row['actual_optimizer_writes']=99; write(f.state/'rank-3.pt',row)
                else:
                    f.driver['candidate_stream']['consumed_source_rows']=10240; write(f.state/'driver.pt',f.driver)
                f.refresh_checkpoint()
                with self.assertRaises(ValueError): f.create()

    def test_fast_source_rechecks_identity_and_QA_without_reloading_tensors(self):
        with tempfile.TemporaryDirectory() as folder:
            f=Fixture(folder); f.create()
            with patch.object(module,'_load_cpu',side_effect=AssertionError('no second CPU tensor load')):
                module.source(f.output)
            qa=f.base/'logs/formal-inline-development-after.json'; qa.write_text(qa.read_text()+' ')
            with self.assertRaises(ValueError): module.source(f.output)

    def test_bind_preserves_science_and_old_job_prefix_not_same_job(self):
        with tempfile.TemporaryDirectory() as folder:
            f=Fixture(folder); f.create(); child=copy.deepcopy(f.run); child.update(run_id='fresh-target',source_commit='9'*40)
            module.bind(child,f.output)
            self.assertEqual(child['run_id'],f.run_id); self.assertEqual(child['source_commit'],'9'*40)
            b=child['full_state_restore']; self.assertEqual(b['kind'],module.KIND)
            self.assertEqual(b['source_job'],'730226'); self.assertFalse(b['same_job_continuation'])
            self.assertEqual(b['actual_optimizer_writes_per_rank'],100)
            self.assertEqual(child['pretraining_baseline_policy'],'reuse_historical')
            wrong=copy.deepcopy(f.run); wrong['budget']['learning_rate']=2e-6
            with self.assertRaisesRegex(ValueError,'SCIENTIFIC'):module.bind(wrong,f.output)
            wrong=copy.deepcopy(f.run);wrong['actor_micro_batch_per_rank']=32
            with self.assertRaisesRegex(ValueError,'SCOPE'):module.bind(wrong,f.output)

    def test_receipt_counter_or_scheduler_tampering_fails_small_manifest_binding(self):
        for mutation in ('writes','scheduler','phase'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as folder:
                f=Fixture(folder);f.create();receipt=module._json(f.output)
                if mutation=='writes':receipt['actual_optimizer_writes_per_rank']=99
                elif mutation=='scheduler':receipt['ranks'][3]['scheduler_step_count']=7
                else:receipt['post_callback_rng_captured']=True
                write(f.output,receipt)
                with self.assertRaises(ValueError):module.source(f.output)

    def test_only_cp5_training_bind_supported_until_pending_refit_is_wired(self):
        with tempfile.TemporaryDirectory() as folder:
            f=Fixture(folder);f.create();source=list(module.source(f.output));source[1]=dict(source[1],step=50)
            with patch.object(module,'source',return_value=tuple(source)), self.assertRaisesRegex(ValueError,'ONLY_CP5'):
                module.bind(copy.deepcopy(f.run),f.output)

    def test_real_initial_optimizer_header_does_not_require_invented_attempt_index(self):
        with tempfile.TemporaryDirectory() as folder:
            f=Fixture(folder)
            for rank in range(4):
                p=f.base/'logs'/f'optimizer-rank-{rank}.jsonl'
                header=dict(kind='INITIAL_OPTIMIZER',run_id=f.run_id,rank=rank,logical_update=0,optimizer_writes=0)
                p.write_text(json.dumps(header)+'\n'+p.read_text())
            f.create();_,receipt,_,_,_=module.source(f.output)
            for rank in range(4):self.assertIn(f'logs/optimizer-rank-{rank}.jsonl',receipt['historical_files'])

    def test_install_copies_actual_heads_with_original_encoder_and_driver_delegates(self):
        with tempfile.TemporaryDirectory() as folder:
            f=Fixture(folder);f.create();child=copy.deepcopy(f.run);module.bind(child,f.output)
            target=Path(folder)/'target';write(target/'logs/run-manifest.json',child)
            installed=module.install_restored_reward_probe(target,child); live=module._json(target/'artifacts/live/manifest.json')
            self.assertEqual(live['encoder'],str(f.encoder));self.assertTrue(Path(live['commit']).is_file())
            self.assertEqual(Path(live['commit']).parent.name,'heads');self.assertFalse(installed['encoder_weights_copied'])
            self.assertFalse((target/'logs/formal-inline-development-before.json').exists())
            calls=[]
            def restore(trainer,saved,run,**kwargs):
                calls.append(kwargs)
                return dict(completed_outer=5, actual_driver_rng_readback_equal=True,
                    actual_candidate_stream_readback_equal=True, actual_candidate_source_cursor=12800,
                    attempt_progress=saved['attempt_progress'],actual_probe_version=saved['probe_version'],actual_probe_round=0)
            with patch.dict(os.environ,SEARCH_G1_BASE=str(target)), \
                 patch.dict('sys.modules',{'search_g1.training_state':SimpleNamespace(restore_easy2560_driver=restore)}), \
                 patch.object(module,'_load_cpu',side_effect=lambda path:module._json(path)):
                result=module.restore_driver(SimpleNamespace(),f.output)
            self.assertEqual(len(calls),1);self.assertEqual(result['source_step'],5)
            self.assertTrue(result['actual_driver_dataequal']);self.assertFalse(result['post_callback_rng_captured'])
            self.assertFalse(result['actual_cuda_restore_accepted'])
            self.assertTrue((target/'logs/full-state-restored-driver.json').is_file())


if __name__=='__main__':unittest.main()
