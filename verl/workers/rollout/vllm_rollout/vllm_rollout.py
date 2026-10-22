# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""
from typing import List
import os
import copy
import numpy as np
from contextlib import contextmanager
from omegaconf import DictConfig
import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn

from verl import DataProto
from verl.utils.torch_functional import get_eos_mask, pad_sequence_to_length
from verl.workers.rollout.base import BaseRollout
from verl.third_party.vllm import LLM, vllm_version
from verl.third_party.vllm import parallel_state as vllm_ps
from vllm import SamplingParams


def actual_sampling_metadata(params, do_sample):
    """Read effective vLLM parameters, rejecting transformed common8 policy."""
    if type(do_sample) is not bool:
        raise ValueError('common8 sampling mode must be explicit boolean')
    fields = ('n', 'best_of', 'temperature', 'top_p', 'top_k', 'min_p',
              'presence_penalty', 'frequency_penalty', 'repetition_penalty',
              'logprobs', 'prompt_logprobs', 'max_tokens', 'min_tokens',
              'stop', 'stop_token_ids', 'ignore_eos', 'include_stop_str_in_output',
              'skip_special_tokens', 'spaces_between_special_tokens',
              'truncate_prompt_tokens', 'seed', 'detokenize', 'logits_processors',
              'logit_bias', 'allowed_token_ids', 'guided_decoding')
    result = {key: copy.deepcopy(getattr(params, key, None)) for key in fields}
    result['do_sample'] = do_sample
    result['logprob_semantics'] = 'selected_token_logprob_at_actual_sampling_call'
    expected = dict(n=1, temperature=1.0 if do_sample else 0.0,
                    top_p=1.0, top_k=-1, min_p=0.0, presence_penalty=0.0,
                    frequency_penalty=0.0, repetition_penalty=1.0)
    if any(type(result[key]) is bool or result[key] != value for key, value in expected.items()):
        raise ValueError('common8 actual sampler is not the declared neutral policy')
    if result['best_of'] not in (None, 1) or result['logits_processors'] not in (None, []):
        raise ValueError('common8 best-of/logits processors are not neutral')
    if result['logprobs'] != 1 or result['min_tokens'] not in (None, 0):
        raise ValueError('common8 selected probabilities or minimum-token policy invalid')
    if (result['truncate_prompt_tokens'] is not None or result['ignore_eos'] is not False or
            result['logit_bias'] not in (None, {}) or result['allowed_token_ids'] is not None or
            result['guided_decoding'] is not None):
        raise ValueError('common8 effective token/logit transformations are not neutral')
    return result

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


class vLLMRollout(BaseRollout):

    def __init__(self, actor_module: nn.Module, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config
        assert not (not config.enforce_eager and config.free_cache_engine), \
            "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get('tensor_model_parallel_size', 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), \
            "tensor parallel size should be less than or equal to the world size"

        if kwargs.get('train_tp', None) is not None:
            # deployed with megatron
            os.environ['CUDA_TIMER_STREAM_KAFKA_ENABLE'] = '0'
            os.environ['MEGATRON_IMPORT_TIMERS'] = '0'
            train_tp = kwargs.get('train_tp', None)
            num_tp_per_train_tp = train_tp // tensor_parallel_size
            if vllm_version in ('0.4.2', '0.5.4', '0.6.3'):
                vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size,
                                                  num_tp_per_train_tp=num_tp_per_train_tp)

        assert model_hf_config.max_position_embeddings >= config.prompt_length + config.response_length, \
            "model context length should be greater than total sequence length"
        scheduler_options = {}
        self._search_g1_flash3_functions = None
        if config.get('flash_attention_3', False):
            import json
            from pathlib import Path
            from search_g1.easy7b_infra import flags
            from search_g1.vllm_flash3 import install_vllm_flash3
            run = json.loads((Path(os.environ['SEARCH_G1_BASE']) / 'logs/run-manifest.json').read_text())
            if (flags(run).get('flash_attention_3_rollout') is not True or tensor_parallel_size != 1
                    or config.get('block_size') != 256):
                raise ValueError('FA3 rollout requires its bound TP1, block256 lean profile')
            os.environ['VLLM_ATTENTION_BACKEND'] = 'FLASH_ATTN'
            self._search_g1_flash3_functions = install_vllm_flash3(run)
            scheduler_options['block_size'] = 256
        for source, target in (('generation_max_num_seqs', 'max_num_seqs'),
                               ('generation_max_batched_tokens', 'max_num_batched_tokens')):
            value = config.get(source, None)
            if value is not None:
                if type(value) is not int or value <= 0:
                    raise ValueError('Invalid continuous generation scheduler bound')
                scheduler_options[target] = value
        self.inference_engine = LLM(actor_module,
                                    tokenizer=tokenizer,
                                    model_hf_config=model_hf_config,
                                    tensor_parallel_size=tensor_parallel_size,
                                    dtype=config.dtype,
                                    enforce_eager=config.enforce_eager,
                                    gpu_memory_utilization=config.gpu_memory_utilization,
                                    skip_tokenizer_init=False,
                                    max_model_len=config.prompt_length + config.response_length,
                                    load_format=config.load_format,
                                    **scheduler_options)
        scheduler = self.inference_engine.llm_engine.scheduler_config
        for field, value in scheduler_options.items():
            if field == 'block_size':
                if self.inference_engine.llm_engine.cache_config.block_size != value:
                    raise ValueError('Actual FA3 cache block size differs')
                continue
            if getattr(scheduler, field) != value:
                raise ValueError('Actual vLLM scheduler differs from declared phase bound')
        self.generation_scheduler_limits = {field: getattr(scheduler, field)
            for field in ('max_num_seqs', 'max_num_batched_tokens')}

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.offload_model_weights()

        kwargs = dict(
            n=1,
            logprobs=1,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        # we may detokenize the result all together later
        if vllm_version in ('0.4.2', '0.5.4', '0.6.3'):
            kwargs['detokenize'] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self._generation_phase = None
        self._generation_phase_setup_handle = None
        self._generation_cache_setup_started = False

    def begin_generation_phase(self, snapshot_meta):
        if (self._generation_phase is not None or self._generation_phase_setup_handle is not None or vllm_version != '0.6.3' or
                self.config.n != 1 or self.config.tensor_model_parallel_size != 1):
            raise ValueError('Continuous generation requires one idle vLLM0.6.3 TP1 engine')
        from reproduction.common8.sampling_capture import RequestCapture, ContinuousRequestService
        from verl.third_party.vllm.vllm_v_0_6_3.llm import capture_request_behavior
        engine = self.inference_engine.llm_engine
        if getattr(engine, '_common8_stop_capture', None) is not None or engine.has_unfinished_requests():
            raise ValueError('Engine has requests or capture outside the declared phase')
        self._generation_phase_setup_handle = {key: snapshot_meta[key] for key in ('phase_id', 'policy_snapshot_id')}
        if self.config.free_cache_engine:
            self._generation_cache_setup_started = True
            self.inference_engine.init_cache_engine()
        capture = RequestCapture(engine)
        self._generation_phase = dict(identity=copy.deepcopy(snapshot_meta), capture=capture,
            service=ContinuousRequestService(engine, capture, capture_request_behavior),
            trajectory_ids=set(snapshot_meta['behavior_identity']['trajectory_ids']))
        engine._common8_stop_capture = capture

    def advance_generation_phase(self, handle, new_requests, max_steps=8):
        phase = self._generation_phase
        if phase is None or handle != {key: phase['identity'][key] for key in ('phase_id', 'policy_snapshot_id')}:
            raise ValueError('Generation request is outside its frozen phase')
        if type(new_requests) is not list:
            raise ValueError('Generation submissions must be an explicit list')
        for request in new_requests:
            if (type(request) is not dict or request.get('policy_snapshot_id') != handle['policy_snapshot_id'] or
                    type(request.get('rollout_ordinal')) is not int or request['rollout_ordinal'] < 0):
                raise ValueError('Generation request snapshot/ordinal differs')
            for name in ('request_id', 'row_id', 'rollout_call_id', 'question_id'):
                if type(request.get(name)) is not str or not request[name]:
                    raise ValueError('Missing independent generation identity: '+name)
            if request['row_id'] not in phase['trajectory_ids']:
                raise ValueError('Request trajectory is absent from its frozen phase')
            tokens = request['prompt_token_ids']
            if (type(tokens) is not list or not tokens or any(type(token) is not int or token < 0 for token in tokens) or
                    type(request.get('response_index')) is not int or request['response_index'] not in range(5)):
                raise ValueError('Invalid trajectory prompt or response identity')
            meta = request['meta_info']
            do_sample = meta.get('do_sample')
            if type(do_sample) is not bool or meta.get('recompute_log_prob') is not False:
                raise ValueError('Phase requests require explicit sampling mode and captured old probabilities')
            if meta.get('validate', False) and do_sample:
                raise ValueError('Validation generation requires actual greedy caller')
            params = copy.deepcopy(self.sampling_params)
            maximum = request.get('max_tokens', meta.get('max_tokens', self.config.response_length))
            if type(maximum) is not int or not 1 <= maximum <= self.config.response_length:
                raise ValueError('Request completion budget exceeds the declared rollout')
            params.max_tokens = maximum
            if not do_sample:
                for key, value in dict(best_of=1, top_p=1.0, top_k=-1, min_p=0.0, temperature=0.0, n=1).items():
                    setattr(params, key, value)
            metadata = actual_sampling_metadata(params, do_sample)
            context = copy.deepcopy(request)
            context['sampling'] = metadata
            phase['service'].submit(request['request_id'], tokens, params, metadata, context)
        completed = []
        for result in phase['service'].advance(max_steps):
            completed.append(dict(request_id=result['request_id'], output=self._phase_request_output(result)))
        return dict(completed=completed, pending_request_ids=list(phase['service'].pending))

    def _phase_request_output(self, result):
        context, row = result['context'], result['behavior']
        if row['actual_request_token_ids'] != context['prompt_token_ids']:
            raise ValueError('Actual asynchronous request tokens differ')
        row.update(schema_version=1, rollout_call_id=context['rollout_call_id'], rollout_ordinal=context['rollout_ordinal'],
            row_id=context['row_id'], policy_snapshot_id=context['policy_snapshot_id'], sampling=context['sampling'],
            eos_token_id=self.eos_token_id, pad_token_id=self.pad_token_id)
        count = row['raw_generated_token_count']
        length = self.config.response_length
        if count > length:
            raise ValueError('Actual asynchronous completion exceeds response budget')
        prompt = torch.tensor([context['prompt_token_ids']], dtype=torch.long)
        response = torch.full((1, length), self.pad_token_id, dtype=torch.long)
        response[0, :count] = torch.tensor(row['raw_generated_token_ids'], dtype=torch.long)
        sampled = torch.zeros((1, length), dtype=torch.bool)
        sampled[0, :count] = True
        probabilities = torch.zeros((1, length), dtype=torch.float32)
        probabilities[0, :count] = torch.tensor(row['selected_logprobs'], dtype=torch.float32)
        attention = torch.cat((torch.ones_like(prompt), get_eos_mask(response, self.eos_token_id, torch.long)), dim=-1)
        positions = torch.arange(prompt.shape[1]+length, dtype=torch.long).unsqueeze(0)
        batch = TensorDict(dict(prompts=prompt, responses=response, input_ids=torch.cat((prompt, response), dim=-1),
            attention_mask=attention, position_ids=positions, sampling_log_probs=probabilities, sampling_mask=sampled), batch_size=1)
        return DataProto(batch=batch, non_tensor_batch={'sampling_behavior': np.array([row], dtype=object)})

    def end_generation_phase(self, handle, abort=False):
        phase = self._generation_phase
        if handle != self._generation_phase_setup_handle or (phase is None and not abort):
            raise ValueError('Closing a different sampling phase')
        result = (phase['service'].close(abort=abort) if phase is not None
                  else dict(completed_requests=0, engine_steps=0, partial_setup_aborted=True))
        result['scheduler_limits'] = copy.deepcopy(self.generation_scheduler_limits)
        if getattr(self, '_search_g1_flash3_functions', None) is not None:
            result['flash3_execution'] = self._search_g1_flash3_functions.report()
        self.inference_engine.llm_engine._common8_stop_capture = None
        if self._generation_cache_setup_started:
            self.inference_engine.free_cache_engine()
            self._generation_cache_setup_started = False
        self._generation_phase = None
        self._generation_phase_setup_handle = None
        return result

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        if self._generation_phase is not None:
            raise ValueError('Legacy batch generation cannot enter an active continuous phase')
        common8 = os.environ.get('SEARCH_G1_PROFILE') == 'common8'
        smoke = prompts.meta_info.get('common8_model_smoke') is True
        if common8 and smoke and prompts.meta_info.get('do_sample') is not False:
            raise ValueError('common8 model-smoke exception requires explicit greedy mode')
        common8 = common8 and not smoke
        if common8:
            call_id = prompts.meta_info.get('behavior_call_id')
            ordinal = prompts.meta_info.get('behavior_rollout_ordinal')
            snapshot = prompts.meta_info.get('policy_snapshot_id')
            row_ids = prompts.non_tensor_batch.get('behavior_row_id')
            if (not isinstance(call_id, str) or not call_id or not isinstance(snapshot, str) or not snapshot or
                    type(ordinal) is not int or ordinal < 0 or
                    row_ids is None or len(row_ids) != len(prompts) or
                    any(not isinstance(value, str) or not value for value in row_ids)):
                raise ValueError('Missing row-aligned common8 sampling call identity')
            if self.config.n != 1 or self.config.tensor_model_parallel_size != 1:
                raise ValueError('common8 behavior protocol requires n1/tensor-parallel1')
        # rebuild vllm cache engine
        if self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        idx = prompts.batch['input_ids']  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']

        # used to construct attention_mask
        eos_token_id = prompts.meta_info['eos_token_id']

        batch_size = idx.size(0)

        idx_list = []
        # parse idx from torch.Tensor to List[List[str]]
        for i in range(batch_size):
            idx_list.append(_pre_process_inputs(self.pad_token_id, idx[i]))

        do_sample = prompts.meta_info.get('do_sample', True)
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1  # if greedy, only 1 response
            }

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            sampler_actual = actual_sampling_metadata(self.sampling_params, do_sample) if common8 else None
            prior_capture = getattr(self.inference_engine, 'capture_behavior_trace', False)
            try:
                if common8:
                    self.inference_engine.capture_behavior_trace = True
                    from reproduction.common8.sampling_capture import capture_scope
                    with capture_scope(self.inference_engine.llm_engine, self.sampling_params,
                            sampler_actual, call_id, list(row_ids), idx_list) as actual_groups:
                        output = self.inference_engine.generate(
                            prompts=None, sampling_params=self.sampling_params,
                            prompt_token_ids=idx_list, use_tqdm=False)
                        if len(output) != 3:
                            raise ValueError('Missing actual pre-padding sampling behavior')
                        actual_groups.attach(output[2])
                else:
                    output = self.inference_engine.generate(
                        prompts=None,  # because we have already convert it to prompt token id
                        sampling_params=self.sampling_params,
                        prompt_token_ids=idx_list,
                        use_tqdm=False)
            finally:
                if common8:
                    self.inference_engine.capture_behavior_trace = prior_capture

        behavior = None
        if common8:
            if len(output) != 3 or len(output[2]) != batch_size:
                raise ValueError('Missing actual pre-padding sampling behavior')
            behavior = output[2]
            for i, row in enumerate(behavior):
                if row['actual_request_token_ids'] != idx_list[i]:
                    raise ValueError('Actual vLLM request prompt differs from submitted tokens')
                row.update(schema_version=1, rollout_call_id=call_id, rollout_ordinal=ordinal,
                           row_id=row_ids[i], policy_snapshot_id=snapshot, sampling=copy.deepcopy(sampler_actual),
                           eos_token_id=eos_token_id, pad_token_id=self.pad_token_id)

        # TODO(sgm): disable logprob when recompute_log_prob is enable
        # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)
        response = output[0].to(idx.device)
        log_probs = output[1].to(idx.device)

        if response.shape[1] < self.config.response_length:
            response = pad_sequence_to_length(response, self.config.response_length, self.pad_token_id)
            log_probs = pad_sequence_to_length(log_probs, self.config.response_length, self.pad_token_id)

        if self.config.n > 1 and do_sample:
            idx = idx.repeat_interleave(self.config.n, dim=0)
            attention_mask = attention_mask.repeat_interleave(self.config.n, dim=0)
            position_ids = position_ids.repeat_interleave(self.config.n, dim=0)
            batch_size = batch_size * self.config.n
        seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_eos_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                'prompts': idx,
                'responses': response,
                'input_ids': seq,  # here input_ids become the whole sentences
                # 'old_log_probs': log_probs, # we will recompute old log prob with actor
                'attention_mask': attention_mask,
                'position_ids': position_ids
            },
            batch_size=batch_size)

        if common8:
            probabilities = torch.zeros(response.shape, dtype=torch.float32, device=response.device)
            sampled_mask = torch.zeros(response.shape, dtype=torch.bool, device=response.device)
            for i, row in enumerate(behavior):
                count = row['raw_generated_token_count']
                if count > response.shape[1] or response[i, :count].tolist() != row['raw_generated_token_ids']:
                    raise ValueError('Padded response differs from actual sampled tokens')
                probabilities[i, :count] = torch.tensor(row['selected_logprobs'], dtype=torch.float32, device=response.device)
                sampled_mask[i, :count] = True
            batch['sampling_log_probs'] = probabilities
            batch['sampling_mask'] = sampled_mask

        # free vllm cache engine
        if self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch, non_tensor_batch={'sampling_behavior': np.array(behavior, dtype=object)} if common8 else {})
