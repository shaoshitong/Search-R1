"""Pause authentic candidate trajectories at the scheduled policy prefix.

An artificial prefix boundary is not an action boundary. Incomplete actions
retain their native chunk records and receive no environment correction until
their original logical action completes. The actor snapshot stays unchanged.
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import copy
from dataclasses import dataclass,field
import time
import uuid

import torch

from search_g1.async_rollout_runtime import AsyncRolloutConfig,RayPhaseService,BoundedRetriever,execute_one_action
from search_g1.prefix20_split import make_turn,logical_sampling
from reproduction.common8.behavior_trace import raw_prefix_to_first_closing_tag
from reproduction.common8.generation_behavior import validated_raw_turn,assemble_output,_left_pad,_observation_ids,_require
from reproduction.common8.context_budget import protocol,final_stage,observation_room


@dataclass
class PrefixState:
    index:int
    trajectory_id:str
    qid:str
    response_index:int
    prompt:list
    ordinal:int=0
    response:list=field(default_factory=list)
    info_mask:list=field(default_factory=list)
    probabilities:list=field(default_factory=list)
    turns:list=field(default_factory=list)
    chunks:list=field(default_factory=list)
    current_raw:list=field(default_factory=list)
    status:str='ready'
    request:dict=None
    cut:dict=None
    final_turn:bool=False
    actual_retrieval:int=0
    valid_actions:int=0
    valid_searches:int=0
    clipped_observations:int=0
    early_final:bool=False

    @property
    def policy_count(self):return sum(self.info_mask)+len(self.current_raw)


class PrefixCollector:
    def __init__(self,manager,identity,states,*,prefix_tokens=80,service_factory=None,tool_runner=None):
        self.prefix_tokens=prefix_tokens
        self.manager=manager;self.identity=identity;self.states=states
        self.options=AsyncRolloutConfig.from_manager(manager)
        self.budget=protocol(manager.config)
        self.service_factory=service_factory or (lambda:RayPhaseService(manager.actor_rollout_wg,4,self.options.engine_step_budget))
        retriever=BoundedRetriever(manager,self.options)
        self.tool_runner=tool_runner or (lambda text,search:execute_one_action(manager,text,search,retriever))
        self.special=sorted(set(int(x) for x in manager.tokenizer.all_special_ids))
        self.meta={};self.sampling=None;self.phase_events=[];self.captured_records=[]

    def prefix_rows(self,states=None):
        rows=[]
        for state in self.states if states is None else states:
            segments=[];tokens=[];positions=[]
            actions=[(turn['turn_index'],turn['native_chunks'][0]['actual_request_token_ids'],
                      turn['prefix_cut']['kept_token_ids']) for turn in state.turns]
            if state.current_raw:
                actions.append((state.ordinal,state.chunks[0]['actual_request_token_ids'],state.current_raw))
            for turn_index,request,policy in actions:
                take=min(self.prefix_tokens-len(tokens),len(policy))
                if take:
                    start=len(tokens);length=len(request)
                    segments.append(dict(turn_index=turn_index,policy_start=start,policy_length=take,
                        position_start=length,request_length=length,context_tokens=request+policy[:take]))
                    tokens.extend(policy[:take]);positions.extend(range(length,length+take))
                if len(tokens)==self.prefix_tokens:break
            if not tokens:raise RuntimeError('PREFIX20_NO_RETAINED_POLICY_TOKEN_AFTER_COMPLETE_ACTION_BUDGET')
            rows.append(dict(trajectory_id=state.trajectory_id,qid=state.qid,response_index=state.response_index,
                initial_prompt_token_ids=state.prompt,policy_token_ids=tokens,
                policy_position_ids=positions,prefix_segments=segments))
        return rows

    def phase(self,states,identity,*,prefix_only):
        manager=self.manager;tokenizer=manager.tokenizer;budget=self.budget
        _require(manager.config.num_gpus==4 and manager.config.max_turns==4,'prefix20_requires_original_four_gpu_recipe')
        phase_id=uuid.uuid4().hex
        service=self.service_factory();opened=False
        unstarted=deque(s for s in states if s.status!='complete');ready=deque();pending={};tools={}
        remaining=len(unstarted);active=0;calls=0;peak=0;environment_actions=0
        limits=min(self.options.agent_workers,self.options.max_inflight_trajectories)
        pool=ThreadPoolExecutor(max_workers=self.options.retrieval_concurrency,thread_name_prefix='prefix20-retrieval')
        started=time.monotonic()

        def resolved(state):
            nonlocal remaining,active
            if state.status=='complete' or prefix_only and state.policy_count>=self.prefix_tokens:
                if state.status!='complete':state.status='paused'
                remaining-=1;active-=1
            else:state.status='ready';ready.append(state)

        def submit(state):
            nonlocal calls
            if not state.chunks:
                state.final_turn=final_stage(len(state.prompt)+len(state.response),state.ordinal,4,budget)
            request=state.prompt+state.response+state.current_raw
            maximum=500-len(state.current_raw)
            if prefix_only:maximum=min(maximum,self.prefix_tokens-state.policy_count)
            _require(maximum>0 and len(request)+maximum<=budget['total_tokens'],'prefix20_invalid_remaining_native_budget')
            rid=phase_id+'-request-'+uuid.uuid4().hex
            call=identity['policy_snapshot_id']+'-call-'+uuid.uuid4().hex
            meta=dict(recompute_log_prob=False,do_sample=True,validate=False,
                behavior_call_id=call,behavior_rollout_ordinal=state.ordinal,
                policy_snapshot_id=identity['policy_snapshot_id'])
            payload=dict(request_id=rid,prompt_token_ids=request,row_id=state.trajectory_id,
                question_id=state.qid,response_index=state.response_index,rollout_ordinal=state.ordinal,
                rollout_call_id=call,policy_snapshot_id=identity['policy_snapshot_id'],max_tokens=maximum,meta_info=meta)
            state.request=payload;state.status='generating';pending[rid]=state
            service.submit(payload,state.index%4);calls+=1

        def complete_action(state,environment):
            nonlocal environment_actions
            environment_actions+=1
            text,done,valid,is_search=environment
            kept=state.cut['kept_token_ids'];request=state.chunks[0]['actual_request_token_ids']
            room=observation_room(len(request),len(kept),budget)
            if state.final_turn:observation=[];clipped=False
            else:
                ids,flags=_observation_ids(manager,[text],[room]);observation=ids[0];clipped=flags[0]
            turn=make_turn(state.chunks,trajectory_id=state.trajectory_id,turn_index=state.ordinal,
                special_token_ids=self.special,prefix_cut=state.cut,
                observation_token_ids=observation if is_search else [],
                correction_token_ids=[] if is_search else observation,final_turn=state.final_turn or bool(done))
            lp=[p for record in state.chunks for p in record['selected_logprobs']]
            state.turns.append(turn);state.response.extend(kept+observation)
            state.probabilities.extend(lp[:len(kept)]+[0.]*len(observation))
            state.info_mask.extend([1]*len(kept)+[0]*len(observation))
            state.actual_retrieval+=int(is_search and not state.final_turn)
            state.valid_searches+=is_search;state.valid_actions+=valid;state.clipped_observations+=int(clipped)
            state.early_final|=bool(state.final_turn and state.ordinal<4)
            state.chunks=[];state.current_raw=[]
            if state.final_turn or done:state.status='complete'
            else:state.ordinal+=1;state.status='ready'
            resolved(state)

        try:
            opened=True
            service.begin(dict(phase_id=phase_id,policy_snapshot_id=identity['policy_snapshot_id'],
                behavior_identity=copy.deepcopy(identity),expected_trajectory_count=len(states)))
            while remaining:
                while unstarted and active<limits:
                    state=unstarted.popleft();state.status='ready';ready.append(state);active+=1
                peak=max(peak,active)
                while ready:submit(ready.popleft())
                for value in service.poll(timeout=.01):
                    state=pending.pop(value['request_id']);output=value['output']
                    _require(len(output)==1 and len(output.non_tensor_batch['sampling_behavior'])==1,
                             'prefix20_native_single_row_return_changed')
                    record=output.non_tensor_batch['sampling_behavior'][0]
                    self.captured_records.append(copy.deepcopy(record))
                    _require(record['request_id']==value['request_id'] and record['row_id']==state.trajectory_id,
                             'prefix20_native_request_identity_changed')
                    validated_raw_turn(manager,output,0,record,request_tokens=state.request['prompt_token_ids'],
                        row_id=state.trajectory_id,identity=identity,call_id=state.request['rollout_call_id'],
                        ordinal=state.ordinal,sampling=record['sampling'])
                    policy=logical_sampling(record['sampling'])
                    if self.sampling is None:self.sampling=policy
                    _require(policy==self.sampling,'prefix20_actual_sampling_policy_changed')
                    self.meta.update(output.meta_info)
                    state.chunks.append(copy.deepcopy(record));state.current_raw.extend(record['raw_generated_token_ids'])
                    stops=record['effective_stop_provenance']['effective_all_stop_token_ids']
                    state.cut=raw_prefix_to_first_closing_tag(state.current_raw,
                        lambda ids:tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False),
                        eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.pad_token_id,special_token_ids=self.special,
                        executor_decode=lambda ids:tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False),
                        effective_stop_token_ids=stops)
                    ended=(state.cut['closing_tag'] is not None or record['finish_reason']=='stop' or len(state.current_raw)==500)
                    if not ended:
                        _require(prefix_only and state.policy_count==self.prefix_tokens,'prefix20_native_pause_is_not_action_completion')
                        state.status='paused';resolved(state);continue
                    actions,_=manager.postprocess_predictions([state.cut['execution_text']])
                    if actions[0]=='search' and not state.final_turn:
                        state.status='tool_pending';tools[pool.submit(self.tool_runner,state.cut['execution_text'],True)]=state
                    else:complete_action(state,self.tool_runner(state.cut['execution_text'],not state.final_turn))
                for future in [f for f in tools if f.done()]:
                    state=tools.pop(future);complete_action(state,future.result())
            _require(not pending and not tools and not ready and not unstarted,'prefix20_phase_has_unresolved_work')
            metrics=service.end(abort=False);opened=False
        except BaseException:
            if opened:service.end(abort=True)
            raise
        finally:pool.shutdown(wait=False,cancel_futures=True)
        event=dict(phase='candidate_prefix' if prefix_only else 'selected_continuation',phase_id=phase_id,
            trajectories=len(states),generation_requests=calls,peak_inflight_trajectories=peak,
            environment_actions=environment_actions,retrieval_concurrency=self.options.retrieval_concurrency,
            phase_elapsed_seconds=time.monotonic()-started,phase_metrics=metrics,actor_updates_during_phase=0)
        self.phase_events.append(event)
        return event

    def output(self,states,identity):
        _require(len(states)>0 and len(states)%5==0 and all(s.status=='complete' for s in states),'dynrpo_complete_G5_required')
        padded=_left_pad([s.prompt for s in states],self.manager.tokenizer.pad_token_id)
        meta=dict(self.meta,async_rollout=dict(role='dynrpo7b_pause_select_continue',
            completed_trajectories=len(states),phases=self.phase_events,actor_updates_during_generation=0,
            generation_requests=sum(e['generation_requests'] for e in self.phase_events),
            environment_actions=sum(e['environment_actions'] for e in self.phase_events),
            peak_inflight_trajectories=max(e['peak_inflight_trajectories'] for e in self.phase_events),
            retrieval_concurrency=self.options.retrieval_concurrency,
            phase_elapsed_seconds=sum(e['phase_elapsed_seconds'] for e in self.phase_events)))
        return assemble_output(row_ids=[s.trajectory_id for s in states],identity=identity,budget=self.budget,
            left=padded['input_ids'],attention=padded['attention_mask'],pad=self.manager.tokenizer.pad_token_id,
            responses=[s.response for s in states],probabilities=[s.probabilities for s in states],
            info_masks=[s.info_mask for s in states],turns=[s.turns for s in states],final_meta=meta,sampling=self.sampling,
            clipped_observations=[s.clipped_observations for s in states],early_final_turns=[s.early_final for s in states],
            turns_stats=[min(s.ordinal+1,5) for s in states],active=[False]*len(states),
            valid_actions=[s.valid_actions for s in states],valid_searches=[s.valid_searches for s in states],
            actual_retrieval=[s.actual_retrieval for s in states],active_counts=[len(states),0])
