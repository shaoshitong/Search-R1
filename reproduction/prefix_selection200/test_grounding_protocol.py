"""Narrow CPU fixtures for actual grounding clocks, four-replica adoption and dump protocol."""
import ast
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import grounding
from . import calibration_interaction
from .interaction_protocol import CONTRACT
from .test_interaction_protocol import TOKENIZER
from search_g1.calibration_evidence import invalid_action_feedback, evidence_deleted_prefix, validate_provenance


def actor(outer, state, *, actual=950, new=19):
    ranks = [dict(rank=rank, actual_adam_steps=actual, new_adam_steps=new,
        optimizer_writes_before=actual-new, optimizer_writes_after=actual,
        optimizer_writes_this_attempt=new, opportunities_processed=20, opportunities=20,
        skipped_update_opportunities=20-new, complete_outer=True, full_budget_complete=new == 20,
        automatic_rollback_performed=False, opportunity_statuses=['written']*new+['skipped_guard']*(20-new),
        written_with_post_guard=0) for rank in range(4)]
    return dict(run_id='run', completed_outer=outer, scheduler_completed_outer=outer,
        actor_version='actor%d' % outer, grounding_state=state, grounding_probe_version=state['version'],
        actual_adam_steps_per_rank=[actual]*4, new_adam_steps_per_rank=[new]*4,
        skipped_update_opportunities=[20-new]*4, opportunities_processed=20, complete_outer=True,
        full_budget_complete=new == 20, status='complete', automatic_rollback_performed=False, rank_receipts=ranks)


class GroundingLifecycleTests(unittest.TestCase):
    def fixture(self, root):
        heads = {}
        for name in ('commit_probe.pkl', 'boundary_probe.pkl'):
            path = root / name; path.write_bytes(('fitted-fixture-'+name).encode()); heads[name] = grounding.sha(path)
        ids = ['question%d' % index for index in range(300)]
        config = dict(run_id='run', grounding_refresh_enabled=True, grounding_refit_rounds=[50,100,150],
            grounding_calibration_questions='/data/calibration300.parquet', grounding_calibration_sha256='calibration',
            grounding_calibration_question_ids=ids, grounding_refit_visible_device=6,
            world_size=4, interaction_protocol=CONTRACT, grounding_probe_version='source:0:fit',
            model_files_sha256={'model.safetensors':'official'})
        def manifest(round):
            payload = dict(run_id='source' if round == 0 else 'run', round=round,
                version='source:0:fit' if round == 0 else 'run:%d:fit' % round,
                encoder='/data/official' if round == 0 else '/data/actor%d' % round,
                encoder_sha256={'model.safetensors':'official' if round == 0 else 'actor'},
                commit=str(root/'commit_probe.pkl'), boundary=str(root/'boundary_probe.pkl'), head_sha256=heads,
                label_proof=dict(question_count=300, question_ids=ids), calibration_report=dict(unique_question_count=300))
            if round:
                payload['label_proof']['interaction_protocol'] = CONTRACT
            path = root/('source%d.json'%round); grounding.once(path,payload); return path
        source = manifest(0)
        config.update(grounding_manifest=str(source), grounding_manifest_sha256=grounding.sha(source))
        return config, manifest

    def adopted(self, base, config, state):
        replicas = [dict(rank=rank, ok=True, version=state['version'], round=state['round'], encoder=state['encoder'])
                    for rank in range(4)]
        return grounding.adopt(base, config, state, replicas)

    def test_round0_real_artifact_reuse_and_four_replica_adoption(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); config,_=self.fixture(root); base=root/'new-run'
            state=grounding.initialize(base, config)
            self.assertTrue(state['imported_actual_round0_fit']); self.assertFalse(state['freshly_fitted_in_this_run'])
            self.assertEqual(grounding.sha(state['manifest_path']), config['grounding_manifest_sha256'])
            live=grounding.publish_live(base, config, state); self.assertTrue(live.is_file())
            adopted=self.adopted(base,config,state)
            attempt=grounding.bind_attempt(config,adopted,None,1)
            self.assertEqual(attempt['grounding_manifest'], adopted['manifest_path'])
            self.assertEqual(attempt['grounding_initial_manifest_sha256'], config['grounding_manifest_sha256'])
            self.assertEqual(grounding.initialize(base,config),adopted)
            proof=grounding.read(adopted['adoption_receipt'])
            proof['replicas'].append(dict(proof['replicas'][0],rank=4))
            broken=root/'bad-adoption.json'; grounding.once(broken,proof)
            with self.assertRaisesRegex(ValueError,'four actual'):
                grounding.checked_state(config,dict(adopted,adoption_receipt=str(broken),adoption_receipt_sha256=grounding.sha(broken)))

    def test_guarded_real_write_clock_and_due_refit_next_outer_binding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);config,manifest=self.fixture(root);base=root/'new'
            initial=self.adopted(base,config,grounding.initialize(base,config))
            previous=actor(50,initial,actual=950,new=19)
            self.assertEqual(grounding.actual_actor_boundary(previous,50)['actual_adam_steps_per_rank'],[950]*4)
            with self.assertRaisesRegex(ValueError,'finish before'):
                grounding.bind_attempt(config,initial,previous,51)
            fitted=dict(grounding._manifest_state(config,manifest(50)),source_actor_version='actor50')
            adopted=self.adopted(base,config,fitted)
            attempt=grounding.bind_attempt(config,adopted,previous,51)
            self.assertEqual(attempt['grounding_state']['round'],50)
            self.assertEqual(attempt['grounding_manifest_sha256'],adopted['manifest_sha256'])
            self.assertEqual(grounding.static_binding(attempt),grounding.static_binding(config))
            invalid=copy.deepcopy(previous);invalid['rank_receipts'][0]['optimizer_writes_this_attempt']=20
            with self.assertRaisesRegex(ValueError,'actual complete'):
                grounding.actual_actor_boundary(invalid,50)
            self.assertEqual([grounding.expected_round(value) for value in (0,49,50,99,100,149,150,199,200)],
                             [0,0,50,50,100,100,150,150,150])

    def test_recovery_uses_completed_due_fit_without_regeneration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);config,manifest=self.fixture(root);base=root/'new'
            initial=self.adopted(base,config,grounding.initialize(base,config))
            snapshot=base/'snapshots/grounding50.json';grounding.once(snapshot,initial)
            joint=dict(run_id='run',outer=50,actor=actor(50,initial),grounding=initial,
                grounding_snapshot=str(snapshot),grounding_snapshot_sha256=grounding.sha(snapshot))
            self.assertEqual(grounding.recovery_state(base,config,joint),initial)
            fitted=dict(grounding._manifest_state(config,manifest(50)),source_actor_version='actor50')
            grounding.once(base/'grounding/rounds/round-000050/fit-complete.json',fitted)
            self.assertEqual(grounding.recovery_state(base,config,joint),fitted)
            grounding.once(base/'preparation/launch-config.json',config)
            grounding.once(base/'outers/outer-000050/complete.json',joint)
            with patch.object(grounding.subprocess,'run') as producer:
                self.assertEqual(grounding.fit(base,50),fitted)
                producer.assert_not_called()
            wrong=dict(fitted,source_actor_version='another')
            grounding.publish(base/'grounding/rounds/round-000050/fit-complete.json',wrong)
            with self.assertRaisesRegex(ValueError,'not bound'):
                grounding.recovery_state(base,config,joint)


class GroundingInteractionTests(unittest.TestCase):
    def test_canned_stream_keeps_correction_and_closed_evidence_under_shared_native_protocol(self):
        class Params(SimpleNamespace):
            @property
            def all_stop_token_ids(self): return set(self.stop_token_ids)
        class Engine:
            def __init__(self):
                self.model_config=SimpleNamespace(dtype='torch.bfloat16',max_model_len=4096)
                self.scheduler=[SimpleNamespace(waiting=[],running=[],swapped=[])]
                self.current=None;self.added=[];self.aborted=[]
            def add_request(self,rid,prompt,params):
                plans=[[ord('?')+1000]*500,[1,2,3],[6,7,8]]
                current=dict(rid=rid,prompt=prompt['prompt_token_ids'],params=params,ids=plans[len(self.added)],tick=0)
                self.added.append(current);self.current=current
                self.scheduler[0].waiting.append(SimpleNamespace(request_id=rid,prompt_token_ids=current['prompt'],sampling_params=params))
            def has_unfinished_requests(self):return self.current is not None
            def step(self):
                current=self.current;current['tick']+=1
                invalid=current['ids'][0]>1000
                if current['tick']>2:raise AssertionError('Grounding waited past first complete action')
                ids=current['ids'] if invalid or current['tick']==2 else current['ids'][:1]
                native=SimpleNamespace(token_ids=ids,logprobs=[{token:SimpleNamespace(logprob=-.125)}for token in ids],
                    finish_reason='length' if invalid else None,stop_reason=None)
                result=SimpleNamespace(request_id=current['rid'],prompt_token_ids=current['prompt'],outputs=[native],finished=invalid)
                if invalid:self.current=None;self.scheduler[0].waiting.clear()
                return[result]
            def abort_request(self,rid):self.aborted.append(rid);self.current=None;self.scheduler[0].waiting.clear()
        engine=Engine()
        def execute(manager,text,do_search,retriever):
            if text=='<search>x</search>':
                self.assertTrue(do_search)
                return '<information>'+'evidence '*100+'</information>',False,1,1
            if text=='<answer>ok</answer>':return '',True,1,0
            return invalid_action_feedback(),False,0,0
        fake={
            'vllm':SimpleNamespace(SamplingParams=Params),
            'search_g1.async_rollout_runtime':SimpleNamespace(AsyncRolloutConfig=SimpleNamespace,
                BoundedRetriever=lambda manager,config:SimpleNamespace(config=config),execute_one_action=execute),
            'search_r1.llm_agent.generation':SimpleNamespace(GenerationConfig=SimpleNamespace,LLMGenerationManager=lambda*a,**k:object())}
        with tempfile.TemporaryDirectory() as temporary:
            model=Path(temporary);(model/'generation_config.json').write_text(json.dumps(dict(eos_token_id=0)))
            llm=SimpleNamespace(llm_engine=engine)
            config=dict(run_id='run',actor_version='actor50',grounding_refit_outer=50,calibration_model=str(model),search_url='unused',interaction_protocol=CONTRACT)
            with patch.dict(sys.modules,fake):
                texts,models,calls,evidence,correction=calibration_interaction.rollouts(llm,['official prompt'],TOKENIZER,True,config)
            self.assertEqual(calls,[1]);self.assertEqual(len(engine.aborted),2)
            validate_provenance(texts[0],models[0],evidence[0],correction[0])
            self.assertEqual(len(correction[0]),1);self.assertEqual(len(evidence[0]),1)
            begin,end=evidence[0][0];self.assertTrue(texts[0][begin:end].endswith('</information>'))
            deleted=evidence_deleted_prefix(texts[0],len(texts[0]),models[0],evidence[0],correction[0])
            self.assertIn(invalid_action_feedback(),deleted);self.assertNotIn('evidence ',deleted)
            self.assertEqual(llm._g1_calibration_interaction_stats['protocol'],CONTRACT)
            self.assertEqual(llm._g1_calibration_interaction_stats['actual_generation_requests'],3)
            self.assertTrue(all(request['params'].max_tokens==500 and request['params'].temperature==1. for request in engine.added))

    def test_actual_dump_gen_batch_reserves_counterfactual_output_inside4096(self):
        source=Path(__file__).resolve().parents[2]/'ts_probe_dump.py'
        tree=ast.parse(source.read_text());node=next(node for node in tree.body if isinstance(node,ast.FunctionDef)and node.name=='gen_batch')
        scope=dict(SamplingParams=lambda**kwargs:SimpleNamespace(**kwargs))
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),scope)
        seen=[]
        llm=SimpleNamespace(_g1_calibration_prompt_limit=4096,_g1_calibration_total_context=4096,
            _g1_calibration_tokenizer=TOKENIZER,_g1_calibration_context_stats=[],
            llm_engine=SimpleNamespace(model_config=SimpleNamespace(max_model_len=4096)),
            generate=lambda prompts,params,**kwargs:seen.append((prompts,params))or[SimpleNamespace(outputs=[SimpleNamespace(text='ok')])])
        self.assertEqual(scope['gen_batch'](llm,['x'*4100],32,False,['</answer>']),['ok'])
        self.assertEqual(len(seen[0][0][0]['prompt_token_ids']),4064)
        self.assertEqual(seen[0][1].max_tokens,32)


if __name__=='__main__':unittest.main()
