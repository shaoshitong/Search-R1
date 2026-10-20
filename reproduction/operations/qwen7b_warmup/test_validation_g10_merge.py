"""Small synthetic CPU acceptance tests; no model, real rollout or remote IO."""
import ast
import copy
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from . import validation_g10_merge as m
from .validation_g10_materialize import validation_plan_events


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(m.canonical(value)+b'\n')


def records(path,values):
    path.parent.mkdir(parents=True,exist_ok=True)
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'wt') as f:
        for value in values:f.write(json.dumps(value)+'\n')


def fixture(base,count=8):
    """Eight questions, two events and all four genuine-shape shard trees."""
    code=base.parent/'synthetic-producer-code';code.mkdir()
    source_root=Path(__file__).resolve().parents[3]
    for relative,wanted in [('reproduction/prefix20_online/pool.py','canonical'),
                            ('reproduction/prefix_selection200/rollout.py','seed_for')]:
        source=source_root/relative
        node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==wanted)
        raw=ast.get_source_segment(source.read_text(),node)
        destination=code/relative;destination.parent.mkdir(parents=True,exist_ok=True);destination.write_text(raw+'\n')
    code_hashes={relative:m.proof(code/relative)['sha256'] for relative in
        ('reproduction/prefix20_online/pool.py','reproduction/prefix_selection200/rollout.py')}
    actor=base.parent/'synthetic-actor-complete.json'
    put(actor,dict(status='MIGRATION_ALL_SELECTED_ASSETS_VERIFIED',revision=m.REVISION))
    questions=[dict(qid='q%d'%i,question_hash=hashlib.sha256(('question%d'%i).encode()).hexdigest(),
        canonical_question_sha256='normalized-%d'%i,candidate_ordinal=i,
        initial_prompt_token_ids=[11,12],split='recipe_selection_validation') for i in range(count)]
    plan=validation_plan_events(questions,['eventA','eventB'],['attemptA','attemptB'],19)
    plan_path=base/'planned-g10-and-fixed-g5.json';put(plan_path,plan)
    question_path=base/'inputs/questions.json';put(question_path,dict(role=m.ROLE,rows=questions))
    configs=[]
    for event_index,name in enumerate(('event-A','event-B')):
        event=base/name;entries=[]
        config=dict(base=str(event),validation_role=m.ROLE,model_revision=m.REVISION,evaluation_only=True,
            fitting_authorized=False,actor_updates_authorized=False,actor_optimizer_updates=0,untouched_final=False,
            tuning_allowed=True,capture_prefix_tokens=20,question_max_tokens=273,feature_layer=26,
            fixed_view_precommitted=True,same_generation_protocol_as_training=True,warmup_only=True,fit_epochs=0,
            source_commit=m.PRODUCER_COMMIT,event_index=event_index,run_id=plan['run_ids'][event_index],
            sampling_attempt_id=plan['sampling_attempt_ids'][event_index],seed=1729,
            code_root=str(code),collector_code_files_sha256=code_hashes,actor_complete=m.proof(actor))
        seed_for=m.pure_seed_function(config)
        for shard in range(4):
            own=questions[shard::4];input_path=event/'inputs'/('shard-%02d'%shard)/'batch-000000.json'
            put(input_path,dict(shard=shard,split='recipe_selection_validation',rows=own));entries.append(m.proof(input_path))
            root=event/'sampling'/('shard-%02d'%shard);dest=root/'batches/batch-000000';outer=dest/'outers/outer-000001'
            native=outer/'prefix/rank-0-packed';native.mkdir(parents=True)
            prefixes=[];states=[];index_rows=[]
            for q in own:
                qi=questions.index(q)
                for response,tid in enumerate(plan['planned_G10'][qi]['event_trajectory_ids'][event_index]):
                    pre=dict(run_id=config['run_id'],trajectory_id=tid,qid=q['qid'],actor_version=m.MODEL_ACTOR,
                        question_hash=q['question_hash'],initial_prompt_token_ids=[11,12],policy_token_ids=[21,22],
                        policy_position_ids=[2,3],retained_policy_length=2,terminal=True,
                        prefix_segments=[dict(turn_index=0,policy_start=0,policy_length=2,request_length=2,
                                              position_start=2,context_tokens=[11,12,21,22])],
                        future_information_included=False,split='recipe_selection_validation',response_index=response)
                    pre['prefix_sha256']=hashlib.sha256(m.canonical(pre)).hexdigest()
                    pre.update(feature_file=str(native/'complete.json'),feature_layer=26,
                               feature_dtype='bfloat16_bits_uint16')
                    prefixes.append(pre);states.append(dict(trajectory_id=tid,prefix_sha256=pre['prefix_sha256'],sampling_attempt_id=config['sampling_attempt_id']))
                    index_rows.append(dict(trajectory_id=tid,qid=q['qid'],dimension=3584,question_length=2,
                        policy_length=2,policy_positions=[2,3]))
            # Tiny opaque bytes are synthetic fixtures, not actual feature tensors.
            files={}
            for filename in ('question.bin','delta.bin','policy.bin','prototype.bin'):
                p=native/filename;p.write_bytes(('synthetic '+filename).encode());files[filename]=m.proof(p)
            records(native/'index.jsonl',index_rows);files['index.jsonl']=m.proof(native/'index.jsonl')
            put(native/'complete.json',dict(status='LOSSLESS_NATIVE_FEATURES_COMPLETE',count=len(prefixes),files=files,
                feature_layer=26,dimension=3584,question_max_tokens=273,prefix_tokens=20))
            records(outer/'prefix/rank-0-prefix.jsonl.gz',prefixes);records(outer/'prefix/rank-0-states.jsonl.gz',states)
            common=dict(run_id=config['run_id'],actor_version=m.MODEL_ACTOR,sampling_attempt_id=config['sampling_attempt_id'],
                questions=len(own),trajectories=5*len(own),sampling_contract='actual-synthetic-contract',
                action_boundary='actual-boundary',interaction_protocol='actual-protocol',actual_compute_dtype='torch.bfloat16')
            prefix=dict(common,states=m.proof(outer/'prefix/rank-0-states.jsonl.gz'),prefix=m.proof(outer/'prefix/rank-0-prefix.jsonl.gz'),actor_hidden=m.proof(native/'complete.json'))
            put(outer/'prefix/scan.json',dict(ranks=[prefix]));scan=m.proof(outer/'prefix/scan.json')['sha256']
            put(outer/'prefix/complete.json',dict(ranks=[prefix],scan_sha256=scan))
            fulls=[];ems=[]
            for pre in prefixes:
                tid=pre['trajectory_id'];qid=pre['qid'];qi=int(qid[1:]);response=pre['response_index']
                y=int(response<(qi+event_index)%6)
                seed=seed_for(config['run_id'],1,tid,0,0,1729,sampling_attempt_id=config['sampling_attempt_id'])
                f=dict(run_id=config['run_id'],trajectory_id=tid,qid=qid,policy_snapshot_id=m.MODEL_ACTOR,
                    response_index=response,split='recipe_selection_validation',question_hash=pre['question_hash'],
                    initial_prompt_token_ids=[11,12],prefix_sha256=pre['prefix_sha256'],prefix_scan_sha256=scan,
                    strict_em=y,response_token_ids=[21,22,23],response_info_mask=[1,1,1],behavior_turns=[dict(actual=True)],
                    sampling_journal=[dict(logical_turn_index=0,native_chunk_index=0,seed=seed,sampling=dict(seed=seed))])
                fulls.append(f);ems.append({key:f[key] for key in ('trajectory_id','qid','strict_em','prefix_sha256','prefix_scan_sha256')})
            records(outer/'full/rank-0-full.jsonl.gz',fulls);records(outer/'full/rank-0-own-em.jsonl',ems)
            full=dict(common,full=m.proof(outer/'full/rank-0-full.jsonl.gz'),own_em=m.proof(outer/'full/rank-0-own-em.jsonl'))
            put(outer/'full/complete.json',dict(full,labels=len(fulls),actor_optimizer_updates=0))
            receipt=dict(run_id=config['run_id'],input_sha256=m.proof(input_path)['sha256'],questions=len(own),
                trajectories=5*len(own),actor_optimizer_updates=0,prefix=prefix,full=full)
            put(dest/'complete.json',receipt)
            put(root/'complete.json',dict(status='ALL_ASSIGNED_CANONICAL_G5_COMPLETE',shard=shard,run_id=config['run_id'],
                model_revision=m.REVISION,questions=len(own),trajectories=5*len(own),batches=[str(dest/'complete.json')]))
        manifest=event/'inputs/manifest.json';put(manifest,dict(batches=entries))
        config['immutable_preparation_sources']=[m.proof(plan_path),m.proof(question_path),m.proof(manifest)]+entries
        path=event/'config.json';put(path,config);configs.append(m.proof(path))
    put(base/'materialization.json',dict(status='RECIPE_SELECTION_VALIDATION_TWO_FRESH_G5_PLANS_PREPARED_NOT_GENERATED',
        untouched_final=False,tuning_allowed=True,fitting_on_validation_allowed=False,questions=count,
        planned_real_trajectories=10*count,fixed_G5_plan=m.proof(plan_path),event_configs=configs))
    return plan


def reseal_full(base,mutation,key='full'):
    batch_path=base/'event-A/sampling/shard-00/batches/batch-000000/complete.json';batch=m.read(batch_path)
    path=Path(batch['full'][key]['path']);values=list(m.rows(path));mutation(values);records(path,values)
    batch['full'][key]=m.proof(path)
    full_complete=path.parent/'complete.json';value=m.read(full_complete);value.update(batch['full'])
    put(full_complete,value);put(batch_path,batch)


class ValidationMergeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.base=Path(self.temp.name).resolve()/'materialized'
        self.plan=fixture(self.base)

    def tearDown(self):self.temp.cleanup()

    def merge(self):return m.merge(self.base,synthetic_fixture=True,expected_questions=8)

    def test_real_id_join_fixed_view_and_idempotent_full_reverification(self):
        result=self.merge();self.assertEqual(result['status'],m.STATUS)
        self.assertEqual((result['questions'],result['trajectories']),(8,80))
        groups=m.read(self.base/'bank/groups.json')['validation'];self.assertEqual(len(groups),8)
        fixed=m.read(self.base/'bank/fixed-G5.json')
        self.assertFalse(fixed['selection_used_EM'])
        self.assertEqual([{k:r[k] for k in ('qid','trajectory_ids')} for r in fixed['validation']],self.plan['fixed_G5']['groups'])
        self.assertFalse(m.read(self.base/'bank/plan-freeze.json')['labels_consulted'])
        features=m.read(self.base/'bank/features.json')
        self.assertEqual(len(features),8)
        self.assertTrue(all(isinstance(path,str) for path in features))
        self.assertEqual(set(m.read(self.base/'bank/feature-sources-by-event.json')),{'event-A','event-B'})
        self.assertEqual(m.read(self.base/'bank/fixed-G5-selection.json'),self.plan['fixed_G5'])
        before={p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in (self.base/'bank').glob('*.json')}
        again=self.merge();self.assertTrue(again['existing_complete_freshly_verified'])
        self.assertEqual(before,{p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in (self.base/'bank').glob('*.json')})

    def test_incomplete_shard_refused_before_any_bank_or_freeze(self):
        (self.base/'event-B/sampling/shard-03/complete.json').unlink()
        with self.assertRaisesRegex(ValueError,'Incomplete actual event/shard'):self.merge()
        self.assertFalse((self.base/'bank').exists())

    def test_corrupt_payload_SHA_never_publishes_complete(self):
        path=self.base/'event-A/sampling/shard-00/batches/batch-000000/outers/outer-000001/full/rank-0-own-em.jsonl'
        with path.open('ab') as f:f.write(b' ')
        with self.assertRaisesRegex(ValueError,'Source SHA/size changed'):self.merge()
        self.assertFalse((self.base/'bank/complete.json').exists())
        self.assertTrue((self.base/'bank/plan-freeze.json').exists())

    def test_resealed_own_EM_disagreement_detected(self):
        reseal_full(self.base,lambda rows:rows[0].update(strict_em=1-rows[0]['strict_em']),key='own_em')
        with self.assertRaisesRegex(ValueError,'own strict EM labels disagree'):self.merge()
        self.assertFalse((self.base/'bank/complete.json').exists())

    def test_resealed_wrong_seed_namespace_detected(self):
        def wrong(rows):rows[0]['sampling_journal'][0].update(seed=999,sampling=dict(seed=999))
        reseal_full(self.base,wrong)
        with self.assertRaisesRegex(ValueError,'native seed differs'):self.merge()

    def test_resealed_cross_event_ID_cannot_replace_own_response(self):
        replacement=self.plan['planned_G10'][0]['event_trajectory_ids'][1][0]
        reseal_full(self.base,lambda rows:rows[0].update(trajectory_id=replacement))
        with self.assertRaisesRegex(ValueError,'wrong precommitted ID'):self.merge()

    def test_plan_containing_EM_is_not_a_precommit_even_when_SHA_rebound(self):
        p=self.base/'planned-g10-and-fixed-g5.json';plan=m.read(p);plan['planned_G10'][0]['labels']=[1]*10;put(p,plan)
        material_path=self.base/'materialization.json';material=m.read(material_path);material['fixed_G5_plan']=m.proof(p);put(material_path,material)
        with self.assertRaisesRegex(ValueError,'EM entered the pre-generation plan'):self.merge()
        self.assertFalse((self.base/'bank').exists())

    def test_modified_fixed_G5_choice_is_rejected_even_with_rebound_plan_SHA(self):
        p=self.base/'planned-g10-and-fixed-g5.json';plan=m.read(p)
        fixed=plan['fixed_G5']['groups'][0];original=list(fixed['trajectory_ids'])
        own=next(g for g in plan['planned_G10'] if g['qid']==fixed['qid'])
        fixed['trajectory_ids'][-1]=next(tid for tid in own['trajectory_ids'] if tid not in original)
        put(p,plan)
        material_path=self.base/'materialization.json';material=m.read(material_path);material['fixed_G5_plan']=m.proof(p);put(material_path,material)
        with self.assertRaisesRegex(ValueError,'Fixed validation G5 differs from label-blind precommit'):self.merge()
        self.assertFalse((self.base/'bank').exists())

    def test_native_payload_corruption_detected_without_tensor_loading(self):
        p=self.base/'event-B/sampling/shard-02/batches/batch-000000/outers/outer-000001/prefix/rank-0-packed/policy.bin'
        p.write_bytes(b'corruption')
        with self.assertRaisesRegex(ValueError,'Source SHA/size changed'):self.merge()

    def test_existing_complete_only_verifies_never_overwrites_tampering(self):
        self.merge();complete=self.base/'bank/complete.json';accepted=complete.read_bytes()
        groups=self.base/'bank/groups.json';bad=m.read(groups);bad['validation'][0]['labels'][0]^=1;put(groups,bad)
        with self.assertRaisesRegex(ValueError,'Source SHA/size changed'):self.merge()
        self.assertEqual(complete.read_bytes(),accepted)
        self.assertEqual(m.read(groups),bad)

    def test_actual_default_rejects_local_artifact_paths(self):
        with self.assertRaisesRegex(ValueError,'must remain remote'):m.merge(self.base)

    def test_actual_G10_consumer_interfaces_without_model_or_GPU(self):
        # Use the runner's real functions, not a parallel imitation. The native
        # reader only checks metadata/index/prototype here; get() and model
        # construction are deliberately never invoked.
        from . import probe_g10_train as consumer
        import os
        import torch
        from unittest.mock import patch
        cuda_before=torch.cuda.is_initialized()
        accepted=self.merge();inputs=accepted['consumer_inputs']
        groups=consumer.u.read(consumer.u.checked(inputs['groups']))[inputs['group_split']]
        manifests=consumer.u.read(consumer.u.checked(inputs['features']))
        with patch.dict(os.environ,{'QWEN7B_QUESTION_MAX_TOKENS':'273','SEARCH_G1_B_PREFIX_TOKENS':'20'}):
            reader=consumer.native_reader(manifests)
        try:
            self.assertEqual(len(reader.records),80)
            self.assertFalse(reader.cache)
            self.assertFalse(reader.prototypes)
            plan=consumer.u.read(consumer.u.checked(inputs['fixed_G5_plan']))
            self.assertTrue(all(set(row)=={'qid','trajectory_ids'} for row in plan['groups']))
            selected=consumer.frozen_validation(groups,plan)
            self.assertEqual(selected,m.read(self.base/'bank/fixed-G5.json')['validation'])
            self.assertFalse(plan['summary']['labels_consulted'])
            self.assertEqual(plan['summary']['role'],'validation')
        finally:reader.close()
        self.assertEqual(torch.cuda.is_initialized(),cuda_before)


if __name__=='__main__':unittest.main()
