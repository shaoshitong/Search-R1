"""Opaque CPU fixtures exercise the real chained scalar/archive consumer.

Torch loading/rank tensor validation, canonical parquet reading and selected
trace verification are injected: these scalar/tensor fixtures contain no
generated trajectories. Separate selection tests cover that consumer. No
synthetic bytes claim CUDA/model/optimizer recovery acceptance.
"""
import builtins
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from search_g1 import easy2560_final_checkpoint as module
from search_g1.test_easy2560_restore import Fixture as RankFixture
from reproduction.easy2560_joint4 import final_cpu_acceptance as final
from reproduction.easy2560_joint4.test_authenticated_history import bind_fixture, put
from reproduction.easy2560_joint4.test_final_cpu_acceptance import OPAQUE_SELECTION_FIXTURE_SCOPE


class Fixture:
    def __init__(self, root, tensor_count=435):
        self.tensor_count = tensor_count
        self.old, self.new, _ = bind_fixture(Path(root), final_mode=True)
        f = self.new; self.base = f.base; self.cp = self.base/'checkpoints/search-g1-common8/actor/global_step_200'
        self.output = Path(root)/'final200-receipt.json'
        self.pool = put(self.base/'assets/eligible.json', dict(records=list(range(8))))
        f.run.update(profile='common8', execution_site='hpc3', execution_backend='slurm', question_probe=None,
            selection_candidate_questions=2560, selection_selected_questions=512, selection_fixed_candidate_budget=True,
            selection_eligible_pool_path=str(self.pool), input_sha256={'train.parquet': 'a'*64})
        put(self.base/'logs/run-manifest.json', f.run)
        self.progress = dict(completed_outer=200, cumulative_writes=295, attempt_index=200,
                             aborted_attempts=0, full_budget_outer=5)
        self.version = f.run_id+':150:actual-fixture'
        self.driver = dict(schema_version=1, easy2560_selection=True, selection_mode='mixed_first',
            question_probe=None, run_id=f.run_id, source_commit=f.commit, completed_outer=200,
            attempt_progress=self.progress, probe_version=self.version, probe_round=150, reward_refresh_pending=False,
            rng=dict(torch_cpu=[1], numpy=[2], python=[3]),
            candidate_stream=dict(candidate_order=list(range(8)), candidate_cursor=0, epoch=64000,
                consumed_source_rows=512000, pending_source=None, pending_offset=0,
                prefix20_pool_binding=dict(eligible_questions=8, eligible_pool_sha256=module.saved.digest(self.pool),
                                          canonical_train_sha256='a'*64)))
        state_dir = self.cp/'training_state'; self.rank_states=[]; proofs=[]
        for rank in range(4):
            value=dict(rank=rank, world_size=4, run_id=f.run_id, source_commit=f.commit, completed_outer=200,
                actual_optimizer_writes=295, optimizer_calls=295, optimizer_rollbacks=0, full_budget_outers=5,
                last_attempt_index=200, used_attempt_ids=[f'{i:032x}' for i in range(1,201)],
                policy_rejections=0, scheduler=dict(last_epoch=200,_step_count=201), optimizer={'fixture_cpu_only':True},
                rng={name:[rank] for name in module.RNG_KEYS})
            self.rank_states.append(value); path=put(state_dir/f'rank-{rank}.pt',value)
            proofs.append(dict(rank=rank,path=str(path),run_id=f.run_id,actual_optimizer_writes=295,
                               sha256=module.saved.digest(path)))
        driver=put(state_dir/'driver.pt',self.driver)
        put(state_dir/'manifest.json',dict(completed_outer=200, question_probe=None, worker_files=proofs,
            driver_file=dict(path=str(driver),sha256=module.saved.digest(driver))))
        shards=sorted(p.name for p in self.cp.glob('*.safetensors'))
        put(self.cp/'model.safetensors.index.json', {'weight_map':{f'weight{i}':shards[i%len(shards)] for i in range(tensor_count)}})
        put(self.cp/'config.json', {})
        for i in range(7):put(self.cp/f'tokenizer-accessory-{i}.json',dict(accessory=i))
        heads=self.base/'artifacts/round_150-attempt-real/heads'
        for name in ('commit_probe.pkl','boundary_probe.pkl'):put(heads/name,b'fixture actual round150 '+name.encode())
        report=put(heads/'calibration-report.json',dict(round=150,actual=True))
        encoder=put(self.base/'encoder/model.safetensors',b'fixture saved round150 encoder')
        self.probe=dict(run_id=f.run_id,version=self.version,round=150,encoder=str(encoder.parent),
            encoder_sha256={encoder.name:module.saved.digest(encoder)},
            commit=str(heads/'commit_probe.pkl'),boundary=str(heads/'boundary_probe.pkl'),
            head_sha256={name:module.saved.digest(heads/name) for name in ('commit_probe.pkl','boundary_probe.pkl')},
            calibration_report_sha256=module.saved.digest(report))
        self.probe_path=put(heads.parent/'version-manifest.json',self.probe)
        self.refresh()

    def refresh(self):
        f=self.new;cp=self.cp
        state=json.loads((cp/'training_state/manifest.json').read_bytes())
        for row in state['worker_files']:row['sha256']=module.saved.digest(row['path'])
        state['driver_file']['sha256']=module.saved.digest(state['driver_file']['path'])
        put(cp/'training_state/manifest.json',state)
        complete=dict(step=200,run_id=f.run_id,checkpoint=str(cp),finite_all_saved_tensors=True,
            validated_tensor_count=self.tensor_count,actual_optimizer_writes_per_rank=295,
            files_sha256={str(p.relative_to(cp)):module.saved.digest(p) for p in cp.rglob('*') if p.is_file() and p.name!='complete.json'})
        put(cp/'complete.json',complete)
        for step in final.STEPS[1:]:
            folder=self.base/f'preparation/checkpoint-archives/global_step_{step}'
            snapshot=json.loads((folder/'source-snapshot.json').read_bytes())
            snapshot['files']['metadata/run-manifest.json']=dict(path=str(self.base/'logs/run-manifest.json'),
                bytes=(self.base/'logs/run-manifest.json').stat().st_size,sha256=module.saved.digest(self.base/'logs/run-manifest.json'))
            if step==200:
                for path in cp.rglob('*'):
                    if path.is_file():snapshot['files']['checkpoint/'+str(path.relative_to(cp))]=dict(path=str(path),bytes=path.stat().st_size,sha256=module.saved.digest(path))
                for path in list(self.probe_path.parent.rglob('*'))+[self.base/'encoder/model.safetensors']:
                    if path.is_file():snapshot['files']['grounding/'+str(path.relative_to(self.base))]=dict(path=str(path),bytes=path.stat().st_size,sha256=module.saved.digest(path))
            snapshot['total_bytes']=sum(row['bytes'] for row in snapshot['files'].values())
            put(folder/'source-snapshot.json',snapshot)
            accepted=json.loads((folder/'complete.json').read_bytes())
            accepted.update(snapshot_sha256=module.saved.digest(folder/'source-snapshot.json'),total_bytes=snapshot['total_bytes'],file_count=len(snapshot['files']))
            put(folder/'complete.json',accepted)

    def create(self):
        with patch.object(final,'canonical',side_effect=self.new.load_canonical), \
             patch.object(final,'selection_evidence',return_value=OPAQUE_SELECTION_FIXTURE_SCOPE), \
             patch.object(module.saved,'_load_cpu',side_effect=lambda path:json.loads(Path(path).read_bytes())), \
             patch.object(module.saved,'_validate_rank',side_effect=RankFixture.rank_check):
            return module.create_receipt(self.base,self.cp,self.output)


class FinalCheckpointTests(unittest.TestCase):
    def test_complete339_tensor_receipt_and_source_use_actual_checkpoint_index(self):
        with tempfile.TemporaryDirectory() as root:
            f = Fixture(root, tensor_count=339)
            f.create()
            _, receipt, cp, _, parent = module.source(f.output, rehash_checkpoint=True)
            self.assertEqual(receipt['hf_saved_tensor_count'], 339)
            for count in (338, 435, 0, True):
                with self.subTest(count=count), self.assertRaisesRegex(ValueError, 'FULL_HF_TENSOR_SET'):
                    module._checkpoint_metadata(dict(receipt, hf_saved_tensor_count=count), parent, cp)

    def test_physical8_final_create_and_source_retain_original_physical16_prefix(self):
        with tempfile.TemporaryDirectory() as root:
            f = Fixture(root)
            original_manifest = (f.old.base/'logs/run-manifest.json').read_bytes()
            original_receipt = (f.base/'restore-receipt.json').read_bytes()
            f.new.run['actor_micro_batch_per_rank'] = 8
            put(f.base/'logs/run-manifest.json', f.new.run)
            f.refresh()
            result = f.create()
            with patch.object(module.saved, '_load_cpu', side_effect=AssertionError('source must not load tensors')):
                _, receipt, _, _, parent = module.source(f.output, rehash_checkpoint=True)
            self.assertEqual(result['world_size'], 4)
            self.assertEqual(parent['actor_micro_batch_per_rank'], 8)
            self.assertEqual(parent['actor_logical_micro_batch_per_rank'], 4)
            self.assertEqual(receipt['authenticated_history']['source_job'], '730226')
            self.assertEqual((f.old.base/'logs/run-manifest.json').read_bytes(), original_manifest)
            self.assertEqual((f.base/'restore-receipt.json').read_bytes(), original_receipt)
            original_run = json.loads(original_manifest)
            self.assertEqual(original_run['actor_micro_batch_per_rank'], 16)
            module.saved._scope(original_run)
            with self.assertRaisesRegex(ValueError, 'EXACT_NATIVE4_EASY_SCOPE'):
                module.saved._scope(dict(original_run, actor_micro_batch_per_rank=8))

    def test_final_scope_rejects_upward_or_wrong_logical_and_topology_before_loading(self):
        for key, value in (('actor_micro_batch_per_rank', 32), ('actor_micro_batch_per_rank', True),
                           ('actor_logical_micro_batch_per_rank', 8), ('training_world_size', 8),
                           ('all_correct_virtual_zero', True)):
            with self.subTest(key=key, value=value), tempfile.TemporaryDirectory() as root:
                f = Fixture(root); f.new.run[key] = value
                put(f.base/'logs/run-manifest.json', f.new.run)
                with patch.object(module.saved, '_load_cpu', side_effect=AssertionError('invalid scope must fail before load')):
                    with self.assertRaisesRegex(ValueError, 'EXACT_NATIVE4_EASY_SCOPE'):
                        f.create()
                self.assertFalse(f.output.exists())

    def test_real_two_execution_chain_full_CP200_and_round150_without_training_bind(self):
        with tempfile.TemporaryDirectory() as root:
            f=Fixture(root);result=f.create()
            with patch.object(module.saved,'_load_cpu',side_effect=AssertionError('source may not load tensors')):
                _,r,cp,base,parent=module.source(f.output,rehash_checkpoint=True)
            self.assertEqual((r['status'],r['kind'],r['step'],r['world_size']),(module.STATUS,module.KIND,200,4))
            self.assertEqual(result['file_count'],19);self.assertEqual(r['driver_state']['source_consumed'],512000)
            self.assertEqual(r['reward_probe_round'],150);self.assertEqual(r['source_job'],'730300')
            self.assertEqual(r['authenticated_history']['source_job'],'730226')
            self.assertFalse(r['single_original_job_prefix']);self.assertTrue(r['no_update201'])
            self.assertFalse(r['post_callback_rng_captured']);self.assertFalse(r['actual_cuda_restore_accepted'])
            self.assertFalse(r['actual_model_optimizer_rng_dataequal_accepted'])
            self.assertEqual(r['sampling_rng_stage'],'saved_checkpoint_before_final51713')
            self.assertEqual((cp,base,parent),(f.cp,f.base,f.new.run))

    def test_forged_prefix_identity_and_update201_fail_before_CPU_tensor_reads(self):
        for mutation in ('prefix','extra201','foreign_worker'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as root:
                f=Fixture(root)
                if mutation=='prefix':
                    path=f.base/'restore-receipt.json';r=json.loads(path.read_bytes());r['source_job']='foreign';put(path,r)
                    f.new.run['full_state_restore']['source_receipt_sha256']=module.saved.digest(path)
                    put(f.base/'logs/run-manifest.json',f.new.run)
                elif mutation=='extra201':
                    path=f.base/'logs/actual-updates.jsonl';row=dict(f.new.updates[-1],step=201)
                    with path.open('a') as stream:stream.write(json.dumps(row)+'\n')
                else:
                    path=f.base/'logs/full-state-restored-workers.json';r=json.loads(path.read_bytes());r['ranks'][2]['source_job']='foreign';put(path,r)
                with patch.object(module.saved,'_load_cpu',side_effect=AssertionError('CPU state loading must follow real history validation')):
                    with self.assertRaises(ValueError):f.create()
                self.assertFalse(f.output.exists())

    def test_four_saved_rank_state_RNG_and_driver_counts_are_required(self):
        for mutation in ('rank','rng','driver'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as root:
                f=Fixture(root)
                if mutation=='driver':
                    f.driver['candidate_stream']['consumed_source_rows']=511999;put(f.cp/'training_state/driver.pt',f.driver)
                else:
                    path=f.cp/'training_state/rank-3.pt';value=json.loads(path.read_bytes())
                    if mutation=='rank':value['actual_optimizer_writes']=294
                    else:del value['rng']['request_generators']
                    put(path,value)
                f.refresh()
                with self.assertRaises(ValueError):f.create()
                self.assertFalse(f.output.exists())

    def test_archived_reward_heads_are_required_not_a_new_unarchived_round150(self):
        with tempfile.TemporaryDirectory() as root:
            f=Fixture(root)
            # Correctly hashed new head still cannot replace the accepted archive dependency.
            head=Path(f.probe['commit']);head.write_bytes(b'new unarchived head')
            f.probe['head_sha256']['commit_probe.pkl']=module.saved.digest(head);put(f.probe_path,f.probe)
            with self.assertRaisesRegex(ValueError,'ARCHIVE_REWARD_DEPENDENCY'):f.create()

    def test_cross_node_device_allowed_but_pinned_metadata_and_same_size_SHA_change_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            f=Fixture(root);f.create();r=json.loads(f.output.read_bytes())
            def other_mount(value):
                if isinstance(value,dict):
                    if 'device' in value:value['device']+=1
                    for item in value.values():other_mount(item)
                elif isinstance(value,list):
                    for item in value:other_mount(item)
            other_mount(r);put(f.output,r);module.source(f.output)
            path=f.base/'logs/final-identity.json';raw=path.read_bytes();st=path.stat()
            path.write_bytes(raw.replace(b'"correct": 3610.0',b'"correct": 3611.0'))
            os.utime(path,ns=(st.st_atime_ns,st.st_mtime_ns))
            with self.assertRaisesRegex(ValueError,'PINNED_CONTENT_SHA_CHANGED'):module.source(f.output)

    def test_full_checkpoint_hash_detects_changed_model_bytes_and_source_requires_chain_records(self):
        with tempfile.TemporaryDirectory() as root:
            f=Fixture(root);f.create()
            row=next(row for row in json.loads(f.output.read_bytes())['files'] if row['relative_path'].endswith('.safetensors'))
            path=Path(row['path']);raw=path.read_bytes();st=path.stat()
            path.write_bytes(bytes([raw[0]^1])+raw[1:]);os.utime(path,ns=(st.st_atime_ns,st.st_mtime_ns))
            with self.assertRaisesRegex(ValueError,'PINNED_CONTENT_SHA_CHANGED'):module.source(f.output,rehash_checkpoint=True)
            path.write_bytes(raw);os.utime(path,ns=(st.st_atime_ns,st.st_mtime_ns))
            receipt=json.loads(f.output.read_bytes());del receipt['history_records']['current/logs/actual-updates.jsonl'];put(f.output,receipt)
            with self.assertRaisesRegex(ValueError,'CHAIN_INPUT_RECORDS_MISSING'):module.source(f.output)

    def test_default_source_never_imports_torch_or_reads_large_checkpoint_bodies(self):
        with tempfile.TemporaryDirectory() as root:
            f=Fixture(root);f.create();original_import=builtins.__import__;original_open=Path.open
            def guarded_import(name,*args,**kwargs):
                if name.split('.')[0] in ('torch','transformers','ray','vllm'):raise AssertionError('source imported '+name)
                return original_import(name,*args,**kwargs)
            def guarded_open(path,*args,**kwargs):
                if str(path).startswith(str(f.cp)) and path.suffix in ('.pt','.safetensors'):
                    raise AssertionError('source read checkpoint tensor body')
                return original_open(path,*args,**kwargs)
            with patch('builtins.__import__',side_effect=guarded_import),patch.object(Path,'open',guarded_open):
                module.source(f.output)

    def test_old_checkpoint5_and_same_job_prefix_cannot_be_final_receipts(self):
        with tempfile.TemporaryDirectory() as root:
            f=Fixture(root);f.create();r=json.loads(f.output.read_bytes());r['step']=5;put(f.output,r)
            with self.assertRaisesRegex(ValueError,'RESTORE_ONLY_SCHEMA'):module.source(f.output)
            with self.assertRaisesRegex(ValueError,'EXACT_CP200'):module.create_receipt(f.base,f.old.cp,Path(root)/'other.json')


if __name__=='__main__':unittest.main()
