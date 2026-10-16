"""Real controller / DataProto CPU tests with simulated native sampling only."""
import copy
import re
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from verl import DataProto
from reproduction.common8.test_generation_behavior import ManagerFixture,TokenizerFixture,sampler_fixture
from reproduction.common8.test_sampling_capture import stop_provenance_fixture
from search_g1.prefix20_generation import PrefixCollector,PrefixState
from search_g1.prefix20_probe_worker import validate_rows
from search_g1.prefix20_split import verify_turn
from search_g1.rollout_phase_contract import scalar_metrics


class Tokenizer(TokenizerFixture):
    fragments=dict(TokenizerFixture.fragments,**{})
    fragments.update({40:'x',50:'<answer>A</ans',51:'wer>'})
    def encode(self,text,*,add_special_tokens):
        assert not add_special_tokens and text in ('evidence','correction')
        return [30 if text=='evidence' else 31]


class NativeFixtureService:
    def __init__(self,script):
        self.script=script;self.requests=[];self.pending=[];self.end_calls=[]
    def begin(self,snapshot):self.snapshot=snapshot
    def submit(self,request,rank):
        self.requests.append((copy.deepcopy(request),rank));self.pending.append((request,rank))
    def poll(self,timeout=.01):
        if not self.pending:return []
        request,rank=self.pending.pop()  # Deliberately finish in reverse order.
        raw,finish=self.script(request,rank)
        sampling=sampler_fixture();sampling['max_tokens']=request['max_tokens']
        record=dict(schema_version=1,request_id=request['request_id'],row_id=request['row_id'],
            rollout_ordinal=request['rollout_ordinal'],rollout_call_id=request['rollout_call_id'],
            policy_snapshot_id=request['policy_snapshot_id'],actual_request_token_ids=request['prompt_token_ids'],
            raw_generated_token_ids=list(raw),raw_generated_token_count=len(raw),
            selected_logprobs=[-.25-.125*i for i in range(len(raw))],finish_reason=finish,stop_reason=None,
            eos_token_id=99,pad_token_id=0,sampling=sampling)
        record['effective_stop_provenance']=stop_provenance_fixture(record,run_id='f'*32,
            slurm_job_id='12345',source_sha256={'cpu_fixture.py':'a'*64})
        array=np.empty(1,dtype=object);array[0]=record
        output=DataProto.from_dict(dict(responses=torch.tensor([raw],dtype=torch.long),
            sampling_mask=torch.ones((1,len(raw)),dtype=torch.bool),
            sampling_log_probs=torch.tensor([record['selected_logprobs']],dtype=torch.float32)),
            non_tensors={'sampling_behavior':array})
        return [dict(request_id=request['request_id'],output=output)]
    def end(self,abort=False):
        self.end_calls.append(abort)
        return [dict(rank=r,weight_syncs=1,completed_requests=sum(rank==r for _,rank in self.requests),
            engine_steps=sum(rank==r for _,rank in self.requests)) for r in range(4)]


def fixture(script,count=1,tool=None):
    manager=ManagerFixture();manager.config.num_gpus=4;manager.tokenizer=Tokenizer()
    def parse(texts):
        matches=[re.search(r'<(answer|search)>(.*?)</\1>',t,re.DOTALL) for t in texts]
        return [m.group(1) if m else None for m in matches],[m.group(2) if m else '' for m in matches]
    manager.postprocess_predictions=parse
    states=[PrefixState(i,'t-%d'%i,'q-%d'%(i//5),i%5,[10,11]) for i in range(count)]
    identity=dict(run_id='f'*32,source_sha256={'cpu_fixture.py':'a'*64},slurm_job_id='12345',
        logical_update=1,policy_snapshot_id='cpu-policy0',parent_group_ids=['q-%d'%i for i in range((count+4)//5)],
        trajectory_ids=[s.trajectory_id for s in states],validation=False)
    services=[];tools=[]
    def make_service():
        value=NativeFixtureService(script);services.append(value);return value
    def execute(text,do_search):
        tools.append((text,do_search))
        if tool is not None:return tool(text,do_search)
        if '<answer>' in text:return '',True,1,0
        if '<search>' in text:return 'evidence',False,1,1
        return 'correction',False,0,0
    collector=PrefixCollector(manager,identity,states,service_factory=make_service,tool_runner=execute)
    return collector,states,identity,services,tools


class Prefix20GenerationTests(unittest.TestCase):
    def test_pause_does_not_execute_environment_and_cross_boundary_tag_executes_once(self):
        def script(request,rank):
            return ([40]*19+[50],'length') if request['max_tokens']==20 else ([51,99],'stop')
        collector,states,identity,services,tools=fixture(script)
        collector.phase(states,identity,prefix_only=True)
        self.assertEqual(tools,[]);self.assertEqual(states[0].status,'paused')
        before=copy.deepcopy(collector.prefix_rows());validate_rows(before,labels_required=False)
        collector.phase(states,identity,prefix_only=False)
        self.assertEqual(len(tools),1);self.assertEqual(states[0].status,'complete')
        self.assertEqual(collector.prefix_rows(),before)
        prefix=services[0].requests[0][0];suffix=services[1].requests[0][0]
        self.assertEqual(suffix['prompt_token_ids'],prefix['prompt_token_ids']+[40]*19+[50])
        self.assertEqual((prefix['max_tokens'],suffix['max_tokens']),(20,480))
        request,raw,lp,environment=verify_turn(states[0].turns[0])
        self.assertEqual(raw,[40]*19+[50,51,99]);self.assertEqual(environment,[])
        self.assertEqual(states[0].probabilities,lp[:21])

    def test_search_before_twenty_preserves_context_rope_and_remaining_action_budget(self):
        def script(request,rank):
            if request['rollout_ordinal']==0:return [20,21,99],'stop'
            if request['max_tokens']==18:return [40]*18,'length'
            return [23,24,99],'stop'
        collector,states,identity,services,tools=fixture(script)
        collector.phase(states,identity,prefix_only=True)
        before=copy.deepcopy(collector.prefix_rows());validate_rows(before,labels_required=False)
        self.assertEqual([s['policy_length'] for s in before[0]['prefix_segments']],[2,18])
        self.assertEqual(before[0]['policy_position_ids'],[2,3]+list(range(5,23)))
        self.assertIn(30,before[0]['prefix_segments'][1]['context_tokens'])
        self.assertEqual(len(tools),1)
        collector.phase(states,identity,prefix_only=False)
        self.assertEqual(services[1].requests[0][0]['max_tokens'],482)
        self.assertEqual(states[0].info_mask[:5],[1,1,0,1,1])
        self.assertEqual(collector.prefix_rows(),before)

    def test_empty_eos_correction_then_short_answer_uses_actual_turn_and_no_regeneration(self):
        def script(request,rank):
            return ([99],'stop') if request['rollout_ordinal']==0 else ([23,24,99],'stop')
        collector,states,identity,services,tools=fixture(script)
        collector.phase(states,identity,prefix_only=True)
        row=collector.prefix_rows()[0];validate_rows([row],labels_required=False)
        self.assertEqual(row['policy_token_ids'],[23,24])
        self.assertEqual(row['prefix_segments'][0]['turn_index'],1)
        self.assertEqual(row['prefix_segments'][0]['context_tokens'][:3],[10,11,31])
        collector.phase(states,identity,prefix_only=False)
        self.assertEqual(services[1].requests,[])

    def test_suffix_and_tool_failures_close_actual_phase_and_keep_raw_records(self):
        def broken_suffix(request,rank):
            if request['max_tokens']==20:return [40]*20,'length'
            raise RuntimeError('native-suffix-failure')
        collector,states,identity,services,tools=fixture(broken_suffix)
        collector.phase(states,identity,prefix_only=True)
        with self.assertRaisesRegex(RuntimeError,'suffix'):collector.phase(states,identity,prefix_only=False)
        self.assertEqual(services[1].end_calls,[True]);self.assertEqual(len(collector.captured_records),1)
        def tool_failure(text,do_search):raise RuntimeError('actual-tool-failure')
        collector,states,identity,services,tools=fixture(lambda request,rank:([20,21,99],'stop'),tool=tool_failure)
        with self.assertRaisesRegex(RuntimeError,'tool'):collector.phase(states,identity,prefix_only=True)
        self.assertEqual(services[0].end_calls,[True]);self.assertEqual(len(collector.captured_records),1)

    def test_exact_selected_2560_only_after_out_of_order_candidate_prefixes(self):
        def script(request,rank):
            if int(request['row_id'].split('-')[1])%2:return [23,24,99],'stop'
            return ([40]*19+[50],'length') if request['max_tokens']==20 else ([51,99],'stop')
        collector,states,identity,services,tools=fixture(script,count=2570)
        collector.phase(states,identity,prefix_only=True)
        selected=states[5:2565];chosen=copy.deepcopy(identity)
        chosen['trajectory_ids']=[s.trajectory_id for s in selected]
        chosen['parent_group_ids']=identity['parent_group_ids'][1:513]
        collector.phase(selected,chosen,prefix_only=False)
        output=collector.output(selected,chosen)
        self.assertEqual(len(output),2560)
        self.assertEqual(output.non_tensor_batch['behavior_trajectory_id'].tolist(),chosen['trajectory_ids'])
        self.assertNotIn('t-0',output.non_tensor_batch['behavior_trajectory_id'])
        self.assertEqual({rank for _,rank in services[0].requests},{0,1,2,3})
        metrics=scalar_metrics(output)
        self.assertEqual(metrics['infra/rollout/weight_syncs_per_rank'],2)
        self.assertEqual(metrics['infra/rollout/completed_trajectories'],2560)
        self.assertEqual(metrics['infra/rollout/completed_requests_sum'],3850)
        # Compose the actual controller output with both trainer denominator
        # gates. Only the native filesystem/identity boundary is a CPU adapter;
        # token/position/mask/trace/selected-LP verification executes unchanged.
        from reproduction.common8 import behavior_runtime as behavior
        from reproduction.common8.behavior_trace import document_sha256,BehaviorTraceError
        from reproduction.common8.precision_contract import NATIVE4_BF16_PROFILE,SAMPLING_JOURNAL_POLICY
        run=dict(profile='common8',run_id=identity['run_id'],source_sha256=identity['source_sha256'],
            execution_site='hpc3',execution_backend='slurm',training_world_size=4,actor_micro_batch_per_rank=4,
            precision_profile=NATIVE4_BF16_PROFILE,vllm_gpu_memory_utilization=.1,
            behavior_probability_policy=SAMPLING_JOURNAL_POLICY)
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);(base/'logs').mkdir()
            def private(_base,trace,attempt_id=None):
                return dict(serialization_transformed=False,private_trace_sha256=document_sha256(trace),
                    private_trace_path=str(base/'constructed-cpu-trace.json'),physical_execution_authenticated=False)
            with patch.object(behavior,'_native_run',return_value=(base,run,'12345')), \
                    patch.object(behavior,'write_private_evidence',side_effect=private):
                context=behavior.context_before_forward(output)
                proof=behavior.freeze_after_probability_gate(output,context,None)
                torch.testing.assert_close(output.batch['old_log_probs'],output.batch['sampling_log_probs'])
                self.assertFalse(proof['hf_recomputation_performed'])
                bad=copy.deepcopy(output);bad.batch['sampling_log_probs'][0,0]-=1.
                with self.assertRaises(BehaviorTraceError):behavior.freeze_after_probability_gate(bad,context,None)


if __name__=='__main__':unittest.main()
