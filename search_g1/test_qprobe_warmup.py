"""CPU evidence for full teacher labels, cache tamper rejection and head budget.

Cache tests run on explicitly supplied remote synthetic-fixture storage.
They cannot establish actual CUDA frozen-teacher or restore acceptance.
"""
import copy
import json
import math
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from search_g1.qprobe_warmup import (
    TeacherCache, moments, verify_frozen, teacher_identity, QUESTIONS, CHUNK_SIZE, STATUS,
)
from search_g1.qprobe_features import sha_file


class ScoringProvenanceTests(unittest.TestCase):
    def setUp(self):
        from search_g1.qprobe_runtime import strict_scoring_sources
        self.sources = strict_scoring_sources()
        self.manifest = dict(run_id='old-teacher-run', model_revision='aa8e72537993ba99e69dfaafa59ed015b17504d1',
            training_world_size=4, initial_model_sha256={'config.json': 'a'*64},
            input_sha256={'train.parquet': 'b'*64}, budget={'common_manifest_sha256': 'c'*64},
            retriever_version='immutable-index', source_sha256=self.sources)
        self.settings = dict(feature_spec_sha='d'*64, sampling_config={'temperature': 1.0},
            agent_config={'max_response_length': 500}, max_turns=4, seed=20261013,
            questions=[dict(qid='question', question_hash='e'*64)])
        self.trainer = SimpleNamespace(config=SimpleNamespace(max_turns=4,
            data=SimpleNamespace(max_start_length=2048, max_prompt_length=4096,
                max_response_length=500, max_obs_length=500),
            retriever=SimpleNamespace(topk=3, url='http://127.0.0.1:18010/retrieve')))
        self.output = SimpleNamespace(meta_info={'behavior_sampling': {'temperature': 1.0, 'top_p': 1.0}})

    def test_changed_strict_vector_extractor_invalidates_cache_and_online_protocol(self):
        from search_g1.qprobe_runtime import _contracts
        old_identity = teacher_identity(self.manifest, **self.settings)
        old_protocol = _contracts(self.trainer, self.output, self.manifest)
        changed_sources = dict(self.sources, **{'search_g1/mixed_runtime.py': 'f'*64})
        with patch('search_g1.qprobe_runtime.strict_scoring_sources', return_value=changed_sources):
            # A stale prepared source cannot be asserted as the newly loaded scorer.
            with self.assertRaisesRegex(ValueError, 'ACTUAL_STRICT_SCORER_DIFFERS'):
                teacher_identity(self.manifest, **self.settings)
            new_manifest = dict(self.manifest, source_sha256=changed_sources)
            changed_identity = teacher_identity(new_manifest, **self.settings)
            changed_protocol = _contracts(self.trainer, self.output, new_manifest)
        self.assertNotEqual(old_identity, changed_identity)
        self.assertNotEqual(old_protocol['strict_scoring_contract_sha'], changed_protocol['strict_scoring_contract_sha'])
        self.assertEqual(old_protocol['sampling_contract_sha'], changed_protocol['sampling_contract_sha'])
        self.assertEqual(old_protocol['retriever_version'], changed_protocol['retriever_version'])
        self.assertEqual(old_identity['scorer_sources']['search_g1/mixed_runtime.py'],
            sha_file(Path(__file__).with_name('mixed_runtime.py')))

    def test_identical_official_teacher_namespace_survives_fresh_run_identity(self):
        from search_g1.qprobe_runtime import _contracts
        fresh = dict(self.manifest, run_id='new-fresh-run', checkpoint_dir='/data/new-run/checkpoints',
            initial_model='/data/relocated-official-model', completed_updates=0)
        self.assertEqual(teacher_identity(self.manifest, **self.settings), teacher_identity(fresh, **self.settings))
        old = _contracts(self.trainer, self.output, self.manifest)
        new = _contracts(self.trainer, self.output, fresh)
        for key in ('sampling_contract_sha', 'strict_scoring_contract_sha', 'retriever_version'):
            self.assertEqual(old[key], new[key])
        self.assertNotEqual(old['run_id'], new['run_id'])

    def test_only_warmup_fit_instrumentation_can_reuse_teacher_identity(self):
        old = dict(self.manifest, source_sha256=dict(self.sources,
            **{'search_g1/qprobe_warmup.py':'a'*64,
               'search_g1/async_rollout_runtime.py':'b'*64}))
        changed = dict(old, source_sha256=dict(old['source_sha256'],
            **{'search_g1/qprobe_warmup.py':'c'*64}))
        expected = teacher_identity(old, **self.settings)
        self.assertEqual(expected, teacher_identity(changed, **self.settings,
            compatible_warmup_source_sha256='a'*64))
        changed['source_sha256']['search_g1/async_rollout_runtime.py']='d'*64
        self.assertNotEqual(expected, teacher_identity(changed, **self.settings,
            compatible_warmup_source_sha256='a'*64))

    def test_curve_callback_serializes_with_remote_path_and_rejects_local_output(self):
        from functools import partial
        import cloudpickle
        from search_g1.qprobe_warmup import write_training_curve
        callback=cloudpickle.loads(cloudpickle.dumps(partial(write_training_curve,
            '/Users/not-allowed/curve.jsonl','synthetic-run')))
        with self.assertRaisesRegex(ValueError,'MUST_REMAIN_REMOTE'):
            callback(dict(event='epoch',epoch=0,training_groups=5120))


class MomentTests(unittest.TestCase):
    def test_every_binary_count_and_sample_std(self):
        for k in range(6):
            vector=[1]*k+[0]*(5-k)
            count,mean,std=moments(vector)
            self.assertEqual(count,k);self.assertEqual(mean,k/5)
            self.assertAlmostEqual(std,math.sqrt(sum((x-mean)**2 for x in vector)/4))

    def test_incomplete_or_nonbinary_labels_never_become_wrong_answers(self):
        for vector in [[],[0]*4,[0]*6,[0,0,0,0,2],[False]*5,[float('nan')]*5]:
            with self.assertRaises(ValueError):moments(vector)

    def test_frozen_readback_rejects_missing_rank_parameter_changes_and_writes(self):
        before=[dict(rank=i,parameter_sha256='model',optimizer_sha256='adam',scheduler_sha256='lr',
                     actual_optimizer_writes=0,completed_outer=0) for i in range(8)]
        self.assertTrue(verify_frozen(before,copy.deepcopy(before))['actual_eight_rank_frozen_readback'])
        for key,value in [('parameter_sha256','changed'),('optimizer_sha256','changed'),
                          ('scheduler_sha256','changed'),('actual_optimizer_writes',1),('completed_outer',1)]:
            after=copy.deepcopy(before);after[3][key]=value
            with self.assertRaises(ValueError):verify_frozen(before,after)
        with self.assertRaises(ValueError):verify_frozen(before,before[:-1])

    def test_full_5120_head_fit_keeps_zero_formal_counters_and_all_classes(self):
        import torch
        from search_g1.qprobe_model import OnlineQProbe, QProbeConfig
        threads=torch.get_num_threads();torch.set_num_threads(1)
        try:
            counts=torch.arange(QUESTIONS)%6
            features=torch.nn.functional.one_hot(counts,6).float()
            head=OnlineQProbe(6,QProbeConfig(bootstrap_max_groups_per_outer=QUESTIONS,seed=20261013),
                              feature_spec_sha='a'*64)
            labels=[dict(qid=str(i),question_hash='b'*64,strict_em=[1]*(i%6)+[0]*(5-i%6),
                complete=True,selection_source='bootstrap_all_candidate',source_outer=0,
                policy_snapshot_id='synthetic-frozen-teacher:0',run_id='synthetic-run',attempt_id='warmup',
                sampling_contract_sha='c'*64,strict_scoring_contract_sha='d'*64,retriever_version='e'*64)
                for i in range(QUESTIONS)]
            cold=torch.nn.functional.cross_entropy(head.predict(features)['logits'],counts).item()
            actor_rng=torch.get_rng_state().clone()
            fit=head.bootstrap(labels,features,actor_outer=0,actor_snapshot_id='new-fresh-run:0')
            self.assertEqual(fit['bootstrap_groups'],QUESTIONS)
            self.assertEqual(len(head.replay_labels),QUESTIONS)
            self.assertEqual({x['correct_count'] for x in head.replay_labels},set(range(6)))
            self.assertEqual((head.completed_outer,head.update_count),(0,0))
            self.assertEqual(head.actor_snapshot_id,'new-fresh-run:0')
            self.assertEqual({x['run_id'] for x in head.replay_labels},{'synthetic-run'})
            self.assertEqual({x['source_outer'] for x in head.replay_labels},{0})
            self.assertTrue(head.optimizer.state)
            self.assertTrue(torch.equal(actor_rng,torch.get_rng_state()))
            self.assertLess(torch.nn.functional.cross_entropy(head.predict(features)['logits'],counts).item(),cold)
            with tempfile.TemporaryDirectory() as temp:
                path=Path(temp)/'head.pt'
                receipt=head.save(path,actor_snapshot_id='new-fresh-run:0')
                restored=OnlineQProbe.load(path,expected_actor_snapshot_id='new-fresh-run:0',
                    expected_feature_spec_sha=head.feature_spec_sha,expected_checkpoint_sha=receipt['checkpoint_sha'])
                self.assertEqual((restored.completed_outer,restored.update_count),(0,0))
                self.assertEqual(restored.actor_snapshot_id,'new-fresh-run:0')
                self.assertEqual({x['run_id'] for x in restored.replay_labels},{'synthetic-run'})
        finally:torch.set_num_threads(threads)


@unittest.skipUnless(os.environ.get('QPROBE_TEST_REMOTE_ROOT'),'remote synthetic cache storage required')
class CacheTests(unittest.TestCase):
    def setUp(self):
        parent=Path(os.environ['QPROBE_TEST_REMOTE_ROOT']);parent.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=parent)
        self.root=Path(self.temp.name)
        self.identity=dict(questions=[dict(qid=str(i),question_hash='a'*64) for i in range(QUESTIONS)],teacher='synthetic')
        self.cache=TeacherCache(self.root,self.identity)
        self.contracts=dict(sampling_contract_sha='b'*64,strict_scoring_contract_sha='c'*64,retriever_version='d'*64)

    def tearDown(self):self.cache.close();self.temp.cleanup()

    def make(self,index):
        raw=self.root/('rollout-%02d.pt'%index);raw.write_bytes(b'clearly-marked-synthetic-fixture')
        groups=[]
        for i in range(index*CHUNK_SIZE,(index+1)*CHUNK_SIZE):
            vector=[1]*(i%6)+[0]*(5-i%6);k,m,s=moments(vector)
            groups.append(dict(**self.identity['questions'][i],strict_em=vector,
                correct_count=k,mean=m,std=s,trajectory_ids=['%d-%d'%(i,j) for j in range(5)]))
        return groups,dict(relative_path=raw.name,bytes=raw.stat().st_size,sha256=sha_file(raw))

    def test_complete_5120_all_classes_and_reuse(self):
        for i in range(10):
            groups,raw=self.make(i);self.cache.commit_chunk(i,groups,raw,self.contracts)
        proof,groups,_=self.cache.verify_complete()
        self.assertEqual(proof['status'],STATUS);self.assertEqual(len(groups),5120)
        self.assertEqual(proof['trajectories'],25600)
        self.assertEqual({x['correct_count'] for x in groups},set(range(6)))
        before=(self.root/'complete.json').read_bytes()
        self.assertEqual(self.cache.verify_complete()[0],proof)
        self.assertEqual((self.root/'complete.json').read_bytes(),before)

    def test_tamper_and_partial_pool_hardfail(self):
        groups,raw=self.make(0);self.cache.commit_chunk(0,groups,raw,self.contracts)
        with self.assertRaises(ValueError):self.cache.verify_complete()
        path=self.root/'chunk-00.json';value=json.loads(path.read_text());value['groups'][0]['std']=0.5
        path.write_text(json.dumps(value))
        with self.assertRaises(ValueError):self.cache.read_chunk(0)

    def test_raw_bytes_and_duplicate_trajectory_hardfail(self):
        groups,raw=self.make(0)
        bad=copy.deepcopy(groups);bad[1]['trajectory_ids']=bad[0]['trajectory_ids']
        with self.assertRaises(ValueError):self.cache.commit_chunk(0,bad,raw,self.contracts)
        self.cache.commit_chunk(0,groups,raw,self.contracts)
        (self.root/raw['relative_path']).write_bytes(b'tampered')
        with self.assertRaises(ValueError):self.cache.read_chunk(0)


if __name__=='__main__':unittest.main()
