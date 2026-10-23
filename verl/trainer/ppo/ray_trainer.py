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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import itertools
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict

import re
import json
from collections import defaultdict

import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance

import re
from search_r1.llm_agent.generation import LLMGenerationManager, GenerationConfig

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]


import torch
from verl.utils.torch_functional import masked_mean


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch['info_mask'] if 'info_mask' in data.batch else data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    if 'ref_log_prob' in data.batch.keys():
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                    kl_penalty=kl_penalty)  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'critic/kl': current_kl, 'critic/kl_coeff': beta}

    return data, metrics


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1,
                      *, all_correct_virtual_zero=False):
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == 'gae':
        values = data.batch['values']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch['token_level_rewards']
        advantages, returns = core_algos.compute_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                      values=values,
                                                                      eos_mask=response_mask,
                                                                      gamma=gamma,
                                                                      lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'grpo':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index,
                                                                        strict_em=data.non_tensor_batch.get('prefix20_strict_em'),
                                                                        all_correct_virtual_zero=all_correct_virtual_zero)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data


def reduce_metrics(metrics: dict):
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def _compute_response_info(batch):
    response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-response_length]
    response_mask = batch.batch['attention_mask'][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(batch, use_critic=True):
    # TODO: add response length
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)

    advantages = batch.batch['advantages']
    returns = batch.batch['returns']

    max_response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch['values']
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        'critic/score/mean':
            torch.mean(sequence_score).detach().item(),
        'critic/score/max':
            torch.max(sequence_score).detach().item(),
        'critic/score/min':
            torch.min(sequence_score).detach().item(),
        # reward
        'critic/rewards/mean':
            torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max':
            torch.max(sequence_reward).detach().item(),
        'critic/rewards/min':
            torch.min(sequence_reward).detach().item(),
        # adv
        'critic/advantages/mean':
            torch.mean(valid_adv).detach().item(),
        'critic/advantages/max':
            torch.max(valid_adv).detach().item(),
        'critic/advantages/min':
            torch.min(valid_adv).detach().item(),
        # returns
        'critic/returns/mean':
            torch.mean(valid_returns).detach().item(),
        'critic/returns/max':
            torch.max(valid_returns).detach().item(),
        'critic/returns/min':
            torch.min(valid_returns).detach().item(),
        **({
            # values
            'critic/values/mean': torch.mean(valid_values).detach().item(),
            'critic/values/max': torch.max(valid_values).detach().item(),
            'critic/values/min': torch.min(valid_values).detach().item(),
            # vf explained var
            'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        } if use_critic else {}),

        # response length
        'response_length/mean':
            torch.mean(response_length).detach().item(),
        'response_length/max':
            torch.max(response_length).detach().item(),
        'response_length/min':
            torch.min(response_length).detach().item(),
        'response_length/clip_ratio':
            torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # prompt length
        'prompt_length/mean':
            torch.mean(prompt_length).detach().item(),
        'prompt_length/max':
            torch.max(prompt_length).detach().item(),
        'prompt_length/min':
            torch.min(prompt_length).detach().item(),
        'prompt_length/clip_ratio':
            torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }

    # metrics for actions
    if 'turns_stats' in batch.meta_info:
        metrics['env/number_of_actions/mean'] = float(np.array(batch.meta_info['turns_stats'], dtype=np.int16).mean())
        metrics['env/number_of_actions/max'] = float(np.array(batch.meta_info['turns_stats'], dtype=np.int16).max())
        metrics['env/number_of_actions/min'] = float(np.array(batch.meta_info['turns_stats'], dtype=np.int16).min())
    if 'active_mask' in batch.meta_info:
        metrics['env/finish_ratio'] = 1 - float(np.array(batch.meta_info['active_mask'], dtype=np.int16).mean())
    if 'valid_action_stats' in batch.meta_info:
        metrics['env/number_of_valid_action'] = float(np.array(batch.meta_info['valid_action_stats'], dtype=np.int16).mean())
        metrics['env/ratio_of_valid_action'] = float((np.array(batch.meta_info['valid_action_stats'], dtype=np.int16) / np.array(batch.meta_info['turns_stats'], dtype=np.int16)).mean())
    if 'actual_retrieval_stats' in batch.meta_info:
        calls = np.array(batch.meta_info['actual_retrieval_stats'], dtype=np.int32)
        metrics['env/actual_retrieval_calls/mean'] = float(calls.mean())
        metrics['env/actual_retrieval_calls/total'] = int(calls.sum())
    if 'valid_search_stats' in batch.meta_info:
        metrics['env/number_of_valid_search'] = float(np.array(batch.meta_info['valid_search_stats'], dtype=np.int16).mean())


    return metrics


def compute_timing_metrics(batch, timing_raw):
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info['prompt_length']).item()
    num_response_tokens = torch.sum(response_info['response_length']).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        'gen': num_response_tokens,
        **{
            name: num_overall_tokens for name in ['ref', 'values', 'adv', 'update_critic', 'update_actor', 'rollout']
        },
    }

    return {
        **{
            f'timing_s/{name}': value for name, value in timing_raw.items()
        },
        **{
            f'timing_per_token_ms/{name}': timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys(
            )) & set(timing_raw.keys())
        },
    }


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


class RayPPOTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 reward_fn=None,
                 val_reward_fn=None):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                               target_kl=config.algorithm.kl_ctrl.target_kl,
                                                               horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

        self._create_dataloader()
        self._init_logger()

    def _init_logger(self):
        from verl.utils.tracking import Tracking
        self.logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

    def _create_dataloader(self):
        from torch.utils.data import DataLoader
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
        self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
                                         tokenizer=self.tokenizer,
                                         prompt_key=self.config.data.prompt_key,
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         filter_prompts=True,
                                         return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         truncation='error')
        if self.config.data.train_data_num is not None:
            if self.config.data.train_data_num > len(self.train_dataset.dataframe):
                print(f"[WARNING] training dataset size is smaller than desired size. Using the dataset as the original size {len(self.train_dataset.dataframe)}")
            else:
                self.train_dataset.dataframe = self.train_dataset.dataframe.sample(self.config.data.train_data_num, random_state=42)
        print(f"filtered training dataset size: {len(self.train_dataset.dataframe)}")

        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=self.config.data.train_batch_size,
                                           shuffle=self.config.data.shuffle_train_dataloader,
                                           drop_last=True,
                                           collate_fn=collate_fn)

        self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                       tokenizer=self.tokenizer,
                                       prompt_key=self.config.data.prompt_key,
                                       max_prompt_length=self.config.data.max_prompt_length,
                                       filter_prompts=True,
                                       return_raw_chat=self.config.data.get('return_raw_chat', False),
                                       truncation='error')
        if self.config.data.val_data_num is not None:
            if self.config.data.val_data_num > len(self.val_dataset.dataframe):
                print(f"[WARNING] validation dataset size is smaller than desired size. Using the dataset as the original size {len(self.val_dataset.dataframe)}")
            else:
                self.val_dataset.dataframe = self.val_dataset.dataframe.sample(self.config.data.val_data_num, random_state=42)
        print(f"filtered validation dataset size: {len(self.val_dataset.dataframe)}")
        if os.environ.get('SEARCH_G1_BASE'):
            from pathlib import Path
            from reproduction.profile import bind_validation_dataframe
            self.val_dataset.dataframe = bind_validation_dataframe(
                self.val_dataset.dataframe, Path(os.environ['SEARCH_G1_BASE']))

        self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                         batch_size=self.config.data.val_batch_size,
                                         shuffle=False,
                                         drop_last=False,
                                         collate_fn=collate_fn)

        print(f'Size of train dataloader: {len(self.train_dataloader)}')
        print(f'Size of val dataloader: {len(self.val_dataloader)}')

        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        from verl.utils.weight_continuation import completed_updates
        continuation_k = completed_updates(self.config, total_training_steps)
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.actor_rollout_ref.actor.optim.completed_updates = continuation_k
            self.config.critic.optim.total_training_steps = total_training_steps

    def _validate(self, final=False, inline_development=False):
        """
        The training loop of PPO with global metric computation.
        Accumulates metrics across all batches before computing final statistics.
        """
        import torch
        calibration_phase = None
        if os.environ.get('SEARCH_G1_PROFILE') == 'common8':
            from reproduction.common8.validation_identity import configure_validation
            from reproduction.common8.calibration_runtime import active_plan
            calibration_phase = active_plan()
            if inline_development:
                assert not final and calibration_phase is None
                from reproduction.common8.formal_inline_evaluation import configure_development
                configure_development(self)
            else:
                configure_validation(self, final=final)
        reward_tensor_lst = []
        data_source_lst = []
        question_id_lst = []
        calibration_rows = []
        common8_evaluation_rows = []
        if hasattr(self.val_reward_fn, 'reset_metrics'):
            self.val_reward_fn.reset_metrics()

        gen_config = GenerationConfig(
            max_turns=self.config.max_turns,
            max_start_length=self.config.data.max_start_length,
            max_prompt_length=self.config.data.max_prompt_length,
            max_response_length=self.config.data.max_response_length,
            max_obs_length=self.config.data.max_obs_length,
            num_gpus=self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes,
            no_think_rl=self.config.algorithm.no_think_rl,
            search_url = self.config.retriever.url,
            topk = self.config.retriever.topk,
        )

        from search_g1.rollout_phase_contract import configure_generation
        configure_generation(gen_config,self.config.actor_rollout_ref.rollout)
        # Agent config preparation
        generation_manager = LLMGenerationManager(
            tokenizer=self.tokenizer,
            actor_rollout_wg=self.actor_rollout_wg,
            config=gen_config,
            is_validation = True,
        )

        if not self.config.do_search:
            for test_data in self.val_dataloader:
                test_batch = DataProto.from_single_dict(test_data)
                if os.environ.get('SEARCH_G1_BASE'):
                    question_id_lst.extend(test_batch.non_tensor_batch['__g1_question_id'].tolist())

                # we only do validation on rule-based rm
                if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
                    return {}

                test_gen_batch = test_batch.pop(['input_ids', 'attention_mask', 'position_ids'])
                test_gen_batch.meta_info = {
                    'eos_token_id': self.tokenizer.eos_token_id,
                    'pad_token_id': self.tokenizer.pad_token_id,
                    'recompute_log_prob': False,
                    'do_sample': False,
                    'validate': True,
                }

                # pad to be divisible by dp_size
                test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
                # unpad
                test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
                print('validation generation end')

                test_batch = test_batch.union(test_output_gen_batch)

                # evaluate using reward_function
                # for certain reward function (e.g. sandbox), the generation can overlap with reward
                reward_tensor = self.val_reward_fn(test_batch)

                reward_tensor_lst.append(reward_tensor)
                data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))
        else:
            for batch_dict in self.val_dataloader:
                timing_raw = {}
                test_batch: DataProto = DataProto.from_single_dict(batch_dict)
                behavior_identity = None
                if os.environ.get('SEARCH_G1_PROFILE') == 'common8':
                    from reproduction.common8.behavior_runtime import prepare_identity
                    behavior_identity = prepare_identity(test_batch, self.global_steps,
                                                         repeats=1, validation=True)
                if os.environ.get('SEARCH_G1_BASE'):
                    question_id_lst.extend(test_batch.non_tensor_batch['__g1_question_id'].tolist())
                # test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n_agent, interleave=True)

                test_gen_batch = test_batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
                test_gen_batch.meta_info = {
                    'eos_token_id': self.tokenizer.eos_token_id,
                    'pad_token_id': self.tokenizer.pad_token_id,
                    'recompute_log_prob': False,
                    'do_sample': False,
                    'validate': True,
                }
                if behavior_identity is not None:
                    from reproduction.common8.behavior_runtime import attach_generation_identity
                    attach_generation_identity(test_gen_batch, behavior_identity, do_sample=False)
                with _timer('step', timing_raw):
                    first_input_ids = test_gen_batch.batch['input_ids'][:, -gen_config.max_start_length:].clone()
                    with _timer('gen', timing_raw):
                        generation_manager.timing_raw = timing_raw
                        final_gen_batch_output = generation_manager.run_llm_loop(
                            gen_batch=test_gen_batch,
                            initial_input_ids=first_input_ids,
                        )

                    test_batch = test_batch.union(final_gen_batch_output)

                    if os.environ.get('SEARCH_G1_PROFILE') == 'common8':
                        from reproduction.common8.validation_identity import rows_from_validation_batch
                        common8_evaluation_rows.extend(rows_from_validation_batch(test_batch, self.tokenizer))
                    if calibration_phase is not None:
                        from reproduction.common8.calibration_evaluation import rows_from_batch
                        calibration_rows.extend(rows_from_batch(test_batch, self.tokenizer))

                    if behavior_identity is not None:
                        from reproduction.common8.behavior_runtime import cast_token_fields
                        cast_token_fields(test_batch)
                    else:
                        for key in test_batch.batch.keys():
                            test_batch.batch[key] = test_batch.batch[key].long()

                    # evaluate using reward_function
                    # for certain reward function (e.g. sandbox), the generation can overlap with reward
                    reward_tensor = self.val_reward_fn(test_batch)

                    reward_tensor_lst.append(reward_tensor)
                    data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))

        reward_tensor = torch.cat([rw.sum(-1) for rw in reward_tensor_lst], dim=0).cpu()  # (batch_size,)
        # reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)
        if inline_development:
            from reproduction.common8.formal_inline_evaluation import record_development
            record_development(self.global_steps,common8_evaluation_rows,reward_tensor.tolist())
        if calibration_phase is not None:
            assert self.config.do_search and not final
            from reproduction.common8.calibration_runtime import evaluation
            evaluation(*calibration_phase, self.global_steps, calibration_rows, reward_tensor.tolist())
        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val/test_score/{data_source}'] = np.mean(rewards)
            metric_dict[f'val/evaluated_count/{data_source}'] = len(rewards)

        if hasattr(self.val_reward_fn, 'get_metrics'):
            extra_metrics = self.val_reward_fn.get_metrics()
        else:
            extra_metrics = getattr(self.val_reward_fn, 'last_metrics', None)
        if isinstance(extra_metrics, dict):
            for key, value in extra_metrics.items():
                metric_dict[f'val/{key}'] = value

        if os.environ.get('SEARCH_G1_BASE'):
            from pathlib import Path
            from reproduction.profile import record_validation
            evaluation_kwargs = {}
            if os.environ.get('SEARCH_G1_PROFILE') == 'common8':
                assert self.config.do_search, 'Formal common8 validation uses the actual search agent protocol'
                assert [row['question_id'] for row in common8_evaluation_rows] == question_id_lst
                assert [row['data_source'] for row in common8_evaluation_rows] == data_sources.tolist()
                assert [row['em'] for row in common8_evaluation_rows] == reward_tensor.tolist()
                evaluation_kwargs['f1'] = [row['f1'] for row in common8_evaluation_rows]
                for source in data_source_reward:
                    values = [row['f1'] for row in common8_evaluation_rows if row['data_source']==source]
                    metric_dict[f'val/f1/{source}'] = sum(values)/len(values)
                metric_dict['val/weighted_f1'] = sum(evaluation_kwargs['f1'])/len(evaluation_kwargs['f1'])
            record_validation(Path(os.environ['SEARCH_G1_BASE']),
                self.global_steps - 1 if final else self.global_steps,
                question_id_lst, data_sources.tolist(), reward_tensor.tolist(), final=final, **evaluation_kwargs)
        return metric_dict


    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.config.algorithm.adv_estimator == 'gae':
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls
            self.use_critic = True

        elif self.config.algorithm.adv_estimator == 'grpo':
            self.use_critic = False
        else:
            raise NotImplementedError

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()
        if self.config.trainer.get('full_state_restore_receipt', None):
            from pathlib import Path
            from search_g1.full_state_restore import publish
            proofs = self.actor_rollout_wg.restore_mixed_full_state(self.config.trainer.full_state_restore_receipt)
            from reproduction.common8.topology import world_size
            run = __import__('json').loads((Path(os.environ['SEARCH_G1_BASE'])/'logs/run-manifest.json').read_text())
            size = world_size(run)
            assert len(proofs)==size and sorted(p['rank'] for p in proofs)==list(range(size))
            assert all(p['actual_adam_scheduler_readback_equal'] and p['actual_rng_readback_equal'] for p in proofs)
            publish(Path(os.environ['SEARCH_G1_BASE'])/'logs/full-state-restored-workers.json',
                    dict(source_step=run['full_state_restore']['completed_outer'],
                         actual_optimizer_writes_per_rank=run['full_state_restore']['actual_optimizer_writes_per_rank'],
                         run_id=run['run_id'],source_commit=run['source_commit'],
                         slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                         source_receipt_sha256=run['full_state_restore']['source_receipt_sha256'],
                         actual_world_size=size,ranks=proofs))

    def _save_checkpoint(self):
        actor_local_path = os.path.join(self.config.trainer.default_local_dir, 'actor',
                                        f'global_step_{self.global_steps}')
        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, 'actor')
        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path)
        stream = getattr(self, 'mixed_stream', None)
        if stream is not None:
            from pathlib import Path
            from search_g1.training_state import atomic_save, driver_rng, compact_stream_state
            state_dir = Path(actor_local_path)/'training_state'
            proofs = self.actor_rollout_wg.capture_mixed_state(str(state_dir), checkpoint=True)
            qprobe = getattr(self, 'qprobe_runtime', None)
            qprobe_proof = None
            if qprobe is not None:
                run_id = __import__('json').loads((Path(os.environ['SEARCH_G1_BASE'])/
                                                  'logs/run-manifest.json').read_text())['run_id']
                qprobe_proof = qprobe.save_checkpoint(state_dir, self.global_steps,
                    compact_stream_state(stream.state()), self.reward_fn.expected_probe_version,
                    actor_snapshot_id=run_id+':'+str(self.global_steps))
            run = __import__('json').loads((Path(os.environ['SEARCH_G1_BASE'])/
                                           'logs/run-manifest.json').read_text())
            if run.get('easy2560_selection') is True:
                from search_g1.training_state import capture_easy2560_driver
                assert qprobe_proof is None
                driver_state = capture_easy2560_driver(self, run)
            else:
                driver_state = dict(completed_outer=self.global_steps,
                candidate_stream=compact_stream_state(stream.state()), rng=driver_rng(),
                probe_version=self.reward_fn.expected_probe_version, question_probe=qprobe_proof,
                attempt_progress=(None if getattr(self,'inline_attempt_progress',None) is None else
                    dict(completed_outer=self.inline_attempt_progress.completed_outer,
                         cumulative_writes=self.inline_attempt_progress.cumulative_writes,
                         attempt_index=self.inline_attempt_progress.attempt_index,
                         aborted_attempts=self.inline_attempt_progress.aborted_attempts,
                         full_budget_outer=self.inline_attempt_progress.full_budget_outer)))
            driver = atomic_save(state_dir/'driver.pt', driver_state)
            import json
            with (state_dir/'manifest.json').open('x') as f:
                json.dump(dict(completed_outer=self.global_steps,worker_files=proofs,driver_file=driver,
                    question_probe=qprobe_proof,
                    scope='actual optimizer/scheduler/RNG/candidate stream; model in sibling HF files; reload acceptance pending'),f,indent=2)
                f.flush();os.fsync(f.fileno())

        if self.use_critic:
            critic_local_path = os.path.join(self.config.trainer.default_local_dir, 'critic',
                                             f'global_step_{self.global_steps}')
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, 'critic')
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path)

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = attention_mask.view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    partitions=global_partition_lst,
                                                    prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from search_g1.prefix20_restore_only import deny_training
        deny_training()

        logger = self.logger
        from verl.utils.weight_continuation import completed_updates
        continuation_k = completed_updates(self.config, self.total_training_steps)
        self.global_steps = continuation_k
        calibration_phase = None
        if os.environ.get('SEARCH_G1_PROFILE') == 'common8':
            from reproduction.common8.calibration_runtime import active_plan
            calibration_phase = active_plan()
            if calibration_phase is not None:
                assert continuation_k == 0 and self.total_training_steps == 301
        five_update_records = []
        inline_attempts=None
        easy_restore=False
        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        if os.environ.get('SEARCH_G1_PROFILE')=='common8':
            from reproduction.common8.formal_inline_evaluation import active_run,evaluate_development
            if active_run() is not None:
                restore_receipt=self.config.trainer.get('full_state_restore_receipt',None)
                from pathlib import Path
                run = __import__('json').loads((Path(os.environ['SEARCH_G1_BASE'])/'logs/run-manifest.json').read_text())
                restored_outer = run.get('full_state_restore', {}).get('completed_outer', 0)
                assert (continuation_k,self.global_steps)==((restored_outer,restored_outer) if restore_receipt else (0,0))
                assert self.total_training_steps==201 and len(self.train_dataloader)>0
                assert self.config.trainer.critic_warmup<=1 and not self.use_critic
                from reproduction.common8.recoverable_attempt import AttemptProgress
                inline_attempts=AttemptProgress()
                if restore_receipt:
                    from search_g1.full_state_restore import restore_driver
                    restore_driver(self,restore_receipt)
                    base,run=active_run()
                    easy_restore=run['full_state_restore'].get('kind') in ('easy2560_native4','dynrpo7b_native4')
                    if run['full_state_restore'].get('kind')=='dynrpo7b_native4':
                        pass
                    elif easy_restore:
                        from search_g1.easy2560_restore_runtime import record_development_history
                        record_development_history(base,run)
                    else:
                        from search_g1.full_state_restore import record_original_baseline
                        record_original_baseline(base,run)
                    progress = getattr(self, '_restored_attempt_progress', None)
                    if progress is None:
                        progress = dict(completed_outer=5,cumulative_writes=100,attempt_index=5,aborted_attempts=0)
                    for name in ('completed_outer','cumulative_writes','attempt_index','aborted_attempts'):
                        setattr(inline_attempts,name,progress[name])
                    inline_attempts.full_budget_outer=progress.get('full_budget_outer',
                        progress['completed_outer'] if progress['cumulative_writes']==progress['completed_outer']*20 else 0)
                    # Finish only the canonical dev512-after5 that the failed
                    # allocation never reached. Do not repeat initial/stage64.
                    if run['full_state_restore'].get('kind') in ('question_probe_pair','easy2560_native4','dynrpo7b_native4'):
                        # The completed pair already has its historical evaluation.
                        # Future validation runs at the normal common8 cadence.
                        pass
                    elif run.get('followup_checkpoint_required'):
                        from search_g1.full_state_restore import reuse_development_after5
                        reuse_development_after5(base,run)
                    else:
                        evaluate_development(self)
                else:
                    evaluate_development(self)

        # we start from step 1
        self.global_steps += 1

        # Agent config preparation
        gen_config = GenerationConfig(
            max_turns=self.config.max_turns,
            max_start_length=self.config.data.max_start_length,
            max_prompt_length=self.config.data.max_prompt_length,
            max_response_length=self.config.data.max_response_length,
            max_obs_length=self.config.data.max_obs_length,
            num_gpus=self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes,
            no_think_rl=self.config.algorithm.no_think_rl,
            search_url = self.config.retriever.url,
            topk = self.config.retriever.topk,
        )

        from search_g1.rollout_phase_contract import configure_generation
        configure_generation(gen_config,self.config.actor_rollout_ref.rollout)
        generation_manager = LLMGenerationManager(
            tokenizer=self.tokenizer,
            actor_rollout_wg=self.actor_rollout_wg,
            config=gen_config,
        )

        mixed_stream = None
        selection_mode = self.config.data.get("selection_mode", "legacy")
        if selection_mode not in ("legacy", "uniform", "mixed_first", "question_probe", "prefix20_probe", "dynrpo7b"):
            raise ValueError("Unknown training question selection mode: " + selection_mode)
        if selection_mode in ("question_probe","prefix20_probe","dynrpo7b") and self.config.data.get("mixed_first_selection", False):
            raise ValueError("Question-probe and post-rollout mixed-first are mutually exclusive")
        self.inline_attempt_progress = inline_attempts
        self.qprobe_runtime = None
        if selection_mode in ("mixed_first", "question_probe", "prefix20_probe", "dynrpo7b") or (
                selection_mode == "legacy" and self.config.data.get("mixed_first_selection", False)):
            from search_g1.mixed_runtime import CandidateStream
            if selection_mode in ('question_probe','prefix20_probe','dynrpo7b'):
                from search_g1.qprobe_runtime import QProbeCandidateStream
                stream_cls = QProbeCandidateStream
            else:
                stream_cls = CandidateStream
            if selection_mode=='prefix20_probe':
                from search_g1.prefix20_data import Prefix20CandidateStream
                mixed_stream=Prefix20CandidateStream(self.train_dataloader,self.config.prefix20_probe.eligible_pool_path,
                    restored_state=getattr(self,'_restored_candidate_state',None))
            elif selection_mode=='mixed_first' and self.config.data.get('selection_eligible_pool_path'):
                from search_g1.prefix20_data import Prefix20CandidateStream
                mixed_stream=Prefix20CandidateStream(self.train_dataloader,self.config.data.selection_eligible_pool_path,
                    restored_state=getattr(self,'_restored_candidate_state',None))
            else:
                mixed_stream = stream_cls(self.train_dataloader,
                    restored_state=getattr(self,'_restored_candidate_state',None))
            self.mixed_stream = mixed_stream
            assert inline_attempts is not None and self.config.do_search
            if easy_restore and selection_mode!='dynrpo7b':
                from search_g1.easy2560_restore_runtime import verify_reconstructed_stream, finish_pending_refresh
                verify_reconstructed_stream(self,mixed_stream)
                # Older milestone saves precede the scheduled grounding refit.
                # Resume that real callback once, after exact state readback
                # and before drawing any candidate for the next outer.
                finish_pending_refresh(self, continuation_k)
            if selection_mode in ("question_probe","prefix20_probe","dynrpo7b"):
                from pathlib import Path
                from search_g1.qprobe_runtime import QProbeRuntime
                run = __import__('json').loads((Path(os.environ['SEARCH_G1_BASE'])/
                                               'logs/run-manifest.json').read_text())
                if selection_mode == 'dynrpo7b':
                    from reproduction.dynrpo7b.runtime import ProbeRuntime
                    probe_config=run['question_probe']
                    self.qprobe_runtime=ProbeRuntime(probe_config,os.environ['SEARCH_G1_BASE'],self.actor_rollout_wg)
                elif selection_mode == 'prefix20_probe':
                    from search_g1.prefix20_runtime import Prefix20Runtime
                    probe_config=self.config.prefix20_probe
                    self.qprobe_runtime=Prefix20Runtime(probe_config,os.environ['SEARCH_G1_BASE'],self.actor_rollout_wg)
                else:
                    probe_config=self.config.question_probe
                    self.qprobe_runtime = QProbeRuntime(probe_config,
                                                        os.environ['SEARCH_G1_BASE'], self.tokenizer, worker_group=self.actor_rollout_wg)
                restored = getattr(self, '_restored_qprobe_state', None)
                snapshot = restored['path'] if restored else probe_config.get("snapshot_path")
                if snapshot:
                    restored_state=self.qprobe_runtime.restore(snapshot, run['run_id']+':'+str(continuation_k))
                    if selection_mode=='prefix20_probe':
                        from search_g1.full_state_restore import equal_tree,publish
                        from search_g1.training_state import compact_stream_state,driver_rng
                        equal_tree(restored_state['candidate_state'],compact_stream_state(mixed_stream.state()))
                        equal_tree(self._restored_driver_rng,driver_rng())
                        publish(Path(os.environ['SEARCH_G1_BASE'])/'logs/prefix20-restored-pair.json',dict(
                            completed_outer=continuation_k,actual_probe_rank_readbacks=self.qprobe_runtime.restore_receipts,
                            actual_selector_rng_equal=True,actual_candidate_stream_equal=True,actual_driver_rng_equal=True))
                if selection_mode=='dynrpo7b' and getattr(self,'_dynrpo7b_restored_phase',None):
                    phase=self._dynrpo7b_restored_phase
                    if phase['probe_pending']:
                        self.qprobe_runtime.fit_after_checkpoint(self,run['full_state_restore']['source_checkpoint'])
                    if phase['grounding_pending']:
                        from reproduction.profile import refresh_after_checkpoint
                        self.reward_fn.expected_probe_version=refresh_after_checkpoint(continuation_k,run['full_state_restore']['source_checkpoint'])
                    if phase['probe_pending'] or phase['grounding_pending']:
                        from reproduction.dynrpo7b.rolling import publish_imported_boundary
                        publish_imported_boundary(self,run['full_state_restore']['source_receipt'])
                if self.qprobe_runtime.completed_outer != continuation_k:
                    raise ValueError("Actor and question-probe successful outer counters differ")
                if selection_mode=='question_probe' and continuation_k == 0 and probe_config.get('teacher_warmup_cache'):
                    from search_g1.qprobe_warmup import run as teacher_warmup
                    teacher_warmup(self, generation_manager, self.qprobe_runtime, mixed_stream)

        # start training loop
        for epoch in (itertools.count() if inline_attempts is not None else range(self.config.trainer.total_epochs)):
            for batch_dict in (itertools.repeat(None) if mixed_stream is not None else self.train_dataloader):
                checkpoint = None
                print(f'epoch {epoch}, step {self.global_steps}')
                metrics = {}
                timing_raw = {}
                attempt_result=None
                attempt_identity=inline_attempts.begin() if inline_attempts is not None else None

                if mixed_stream is not None:
                    if selection_mode=='dynrpo7b':
                        from reproduction.dynrpo7b.collection import collect
                    elif selection_mode=='prefix20_probe':
                        from search_g1.prefix20_collection import collect
                    elif self.qprobe_runtime is not None:
                        from search_g1.qprobe_runtime import collect
                    else:
                        from search_g1.mixed_runtime import collect
                    batch, mixed_output, behavior_identity, mixed_audit = collect(
                        self, generation_manager, mixed_stream, attempt_identity)
                    timing_raw["candidate_fill"] = mixed_audit["candidate_fill_seconds"]
                    calibration_parents = None
                    metrics.update({"selection/"+k: mixed_audit[k] for k in
                        ("attempted_candidates","selected_mixed","selected_all_correct_fallback","raw_generated_trajectories", "raw_generated_policy_tokens",
                         "raw_actual_retrieval_calls", "candidate_fill_seconds")})
                    if self.config.data.get('selection_fixed_candidate_budget',False):
                        metrics['selection/selected_mixed_fraction']=mixed_audit['selected_mixed']/mixed_audit['selected_groups']
                        metrics['selection/selected_all_correct_fraction']=mixed_audit['selected_all_correct_fallback']/mixed_audit['selected_groups']
                        metrics['selection/insufficient_pool_retries']=mixed_audit['pool_retry']
                    if selection_mode in ('prefix20_probe','dynrpo7b'):
                        metrics['selection/selected_mixed_fraction']=mixed_audit['selected_mixed_fraction']
                        metrics['selection/completed_selected_trajectories']=mixed_audit['completed_selected_trajectories']
                    elif self.qprobe_runtime is not None:
                        from search_g1.qprobe_driver_metrics import selection_diagnostics
                        metrics.update(selection_diagnostics(mixed_audit))
                else:
                    batch: DataProto = DataProto.from_single_dict(batch_dict)
                    behavior_identity = None
                    if os.environ.get('SEARCH_G1_PROFILE') == 'common8':
                        from reproduction.common8.behavior_runtime import prepare_identity
                        behavior_identity = prepare_identity(batch, self.global_steps, repeats=5,
                            **({key:attempt_identity[key] for key in ('attempt_id','attempt_index')}
                               if attempt_identity is not None else {}))
                    calibration_parents = None
                    if calibration_phase is not None:
                        from reproduction.common8.calibration_runtime import training_inputs
                        calibration_parents = training_inputs(batch, calibration_phase)
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n_agent, interleave=True)
                    if calibration_phase is not None:
                        training_inputs(batch, calibration_phase, calibration_parents)

                    # pop those keys for generation
                    gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
                    if behavior_identity is not None:
                        from reproduction.common8.behavior_runtime import attach_generation_identity
                        attach_generation_identity(gen_batch, behavior_identity, do_sample=True)

                ####################
                # original code here

                with _timer('step', timing_raw):
                    if not self.config.do_search:
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                        batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                                dtype=object)
                        # repeat to align with repeated responses in rollout
                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                        batch = batch.union(gen_batch_output)

                ####################
                # Below is aLL about agents - the "LLM + forloop"
                ####################
                # with _timer('step', timing_raw):
                    else:
                        if mixed_stream is not None:
                            final_gen_batch_output = mixed_output
                        else:
                            first_input_ids = gen_batch.batch['input_ids'][:, -gen_config.max_start_length:].clone().long()

                            with _timer('gen', timing_raw):
                                generation_manager.timing_raw = timing_raw
                                final_gen_batch_output = generation_manager.run_llm_loop(
                                    gen_batch=gen_batch,
                                    initial_input_ids=first_input_ids,
                                )

                        from search_g1.rollout_phase_contract import scalar_metrics as rollout_phase_metrics
                        metrics.update(rollout_phase_metrics(final_gen_batch_output))
                        # final_gen_batch_output.batch.apply(lambda x: x.long(), inplace=True)
                        if behavior_identity is not None:
                            from reproduction.common8.behavior_runtime import (
                                cast_token_fields, context_before_forward, freeze_after_probability_gate,
                            )
                            cast_token_fields(final_gen_batch_output)
                            behavior_context = context_before_forward(final_gen_batch_output)
                        else:
                            for key in final_gen_batch_output.batch.keys():
                                final_gen_batch_output.batch[key] = final_gen_batch_output.batch[key].long()

                        with torch.no_grad():
                            if behavior_identity is not None and behavior_context['sampling_journal_only']:
                                freeze_after_probability_gate(final_gen_batch_output, behavior_context, None)
                            else:
                                output = self.actor_rollout_wg.compute_log_prob(final_gen_batch_output)
                                if behavior_identity is not None:
                                    freeze_after_probability_gate(final_gen_batch_output, behavior_context, output)
                                else:
                                    final_gen_batch_output = final_gen_batch_output.union(output)
                            if behavior_identity is not None:
                                del behavior_context

                        # batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                        #                                         dtype=object)
                        batch.non_tensor_batch['uid'] = batch.non_tensor_batch['index'].copy()

                        # repeat to align with repeated responses in rollout
                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                        batch = batch.union(final_gen_batch_output)

                    ####################
                    ####################

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    # batch.batch.apply(lambda x, key: x.long() if key != "old_log_probs" else x, inplace=True, key=True)
                    if behavior_identity is not None:
                        cast_token_fields(batch)
                    else:
                        for key in batch.batch.keys():
                            if key != 'old_log_probs':
                                batch.batch[key] = batch.batch[key].long()

                    actor_config=self.config.actor_rollout_ref.actor
                    skip_zero_reference=(os.environ.get('SEARCH_G1_PROFILE')=='common8' and
                        actor_config.use_kl_loss and float(actor_config.kl_loss_coef)==0)
                    if self.use_reference_policy and skip_zero_reference:
                        metrics['infra/reference_forward_skipped']=1
                    if self.use_reference_policy and not skip_zero_reference:
                        # compute reference log_prob
                        with _timer('ref', timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer('adv', timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_tensor = self.reward_fn(batch)
                        for key,value in (self.reward_fn.get_metrics() or {}).items():
                            metrics['train/'+key] = value
                        batch.batch['token_level_scores'] = reward_tensor

                        virtual_zero = self.config.algorithm.get('all_correct_virtual_zero', False)
                        if type(virtual_zero) is not bool:
                            raise ValueError('PREFIX20_VIRTUAL_ZERO_FLAG_MUST_BE_BOOLEAN')
                        if virtual_zero:
                            if self.config.algorithm.adv_estimator != 'grpo':
                                raise ValueError('PREFIX20_VIRTUAL_ZERO_REQUIRES_GRPO')
                            from search_g1.prefix20_advantage import strict_em_from_batch
                            # Compute at the balanced row order, independently of soft reward.
                            batch.non_tensor_batch['prefix20_strict_em'] = strict_em_from_batch(batch, self.tokenizer)

                        # compute rewards. apply_kl_penalty if available
                        if not self.config.actor_rollout_ref.actor.use_kl_loss:
                            batch, kl_metrics = apply_kl_penalty(batch,
                                                                 kl_ctrl=self.kl_ctrl,
                                                                 kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        # compute advantages, executed on the driver process
                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  num_repeat=self.config.actor_rollout_ref.rollout.n,
                                                  all_correct_virtual_zero=virtual_zero)

                    if mixed_stream is not None:
                        sources = batch.non_tensor_batch['selection_source']
                        response_width = batch.batch['responses'].shape[-1]
                        response_mask = (batch.batch['loss_mask'] if 'loss_mask' in batch.batch
                            else batch.batch['attention_mask'][:, -response_width:]).bool()
                        rewards = batch.batch['token_level_rewards'].sum(-1)
                        selection_sources = (('predicted_mixed',) if selection_mode in ('prefix20_probe','dynrpo7b') else
                            ('random_exploration','predicted_mixed') if selection_mode=='question_probe' else ('mixed','all_correct_fallback'))
                        for source in selection_sources:
                            rows = torch.as_tensor(sources == source, device=rewards.device)
                            metrics['selection/'+source+'/trajectories'] = int(rows.sum())
                            if bool(rows.any()):
                                metrics['selection/'+source+'/original_reward_mean'] = float(rewards[rows].mean())
                                valid = response_mask[rows]
                                advantages = batch.batch['advantages'][rows]
                                if bool(valid.any()):
                                    metrics['selection/'+source+'/original_advantage_mean'] = float(advantages[valid].mean())
                                    metrics['selection/'+source+'/original_advantage_abs_mean'] = float(advantages[valid].abs().mean())

                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            if self.config.do_search and self.config.actor_rollout_ref.actor.state_masking:
                                batch, metrics = self._create_loss_mask(batch, metrics)
                            if calibration_phase is not None:
                                from reproduction.common8.calibration_runtime import training_groups_before_update
                                training_groups_before_update(batch, calibration_phase)
                            batch.meta_info["logical_update"]=self.global_steps
                            if attempt_identity is not None:
                                batch.meta_info.update(attempt_identity)
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        if inline_attempts is not None:
                            from reproduction.common8.recoverable_attempt import record_attempt
                            attempt_result=inline_attempts.finish(
                                actor_output.non_tensor_batch['recoverable_outer_result'].tolist())
                            record_attempt(attempt_result)
                            if not attempt_result['complete_outer']:
                                if self.qprobe_runtime is not None:
                                    self.qprobe_runtime.abort_outer(self.global_steps, attempt_identity,
                                        reason=attempt_result['status'])
                                # The next dataloader batch is a fresh512/G5
                                # rollout at the retained weights in this process.
                                if inline_attempts.aborted_attempts<=3 or inline_attempts.aborted_attempts%10==0:
                                    print('SEARCH_G1_RECOVERABLE_ATTEMPT_SKIPPED='+json.dumps(attempt_result),flush=True)
                                continue
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)
                        if self.qprobe_runtime is not None:
                            with _timer('question_probe_update', timing_raw):
                                from pathlib import Path
                                run_id = __import__('json').loads((Path(os.environ['SEARCH_G1_BASE'])/
                                    'logs/run-manifest.json').read_text())['run_id']
                                event = self.qprobe_runtime.complete_outer(self.global_steps,
                                    run_id+':'+str(self.global_steps), attempt_identity)
                                for name, value in event['head_update'].items():
                                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                                        metrics['question_probe/'+name] = value

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        (self.global_steps % self.config.trainer.test_freq == 0 or
                        self.global_steps == self.config.trainer.get('acceptance_step',0)):
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and \
                            (self.global_steps % self.config.trainer.save_freq == 0 or
                            self.global_steps == self.config.trainer.get('acceptance_step',0) or
                            (self.config.trainer.get('full_state_restore_receipt',None) and
                             not easy_restore and
                             self.global_steps == min(continuation_k + 5, self.total_training_steps - 1)) or
                            self.global_steps + 1 >= self.total_training_steps):
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()
                            from reproduction.profile import checkpoint_complete
                            checkpoint = os.path.join(self.config.trainer.default_local_dir, 'actor', f'global_step_{self.global_steps}')
                            checkpoint_complete(self.global_steps, checkpoint)
                            if selection_mode=='dynrpo7b':
                                from reproduction.dynrpo7b.rolling import bind_actor_checkpoint
                                bind_actor_checkpoint(self,checkpoint)
                                self.qprobe_runtime.fit_after_checkpoint(self,checkpoint)

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)
                if os.environ.get('SEARCH_G1_PROFILE') == 'common8':
                    from reproduction.common8.update_evidence import record
                    record(self.global_steps, metrics, self.reward_fn.expected_probe_version,
                           **(dict(attempt_result=attempt_result) if inline_attempts is not None else {}))
                if calibration_phase is not None:
                    assert self.global_steps == 1, 'Only one fresh calibration outer is permitted'
                    from reproduction.common8.calibration_runtime import finish_after_update
                    finish_after_update(self, calibration_phase)
                    return
                if continuation_k < self.global_steps <= continuation_k + 5:
                    five_update_records.append(dict(step=self.global_steps, metrics=dict(metrics),
                        probe_version=getattr(self.reward_fn, 'last_probe_version', None)))
                if (self.global_steps == self.config.trainer.get('acceptance_step', 0) and
                        (not self.config.trainer.get('full_state_restore_receipt',None) or
                         (easy_restore and continuation_k < self.global_steps))):
                    if os.environ.get('SEARCH_G1_PROFILE')=='common8':
                        from reproduction.common8.formal_inline_evaluation import active_run,evaluate_development
                        if active_run() is not None:evaluate_development(self)
                    if not self.config.trainer.get('full_state_restore_receipt',None):
                        from reproduction.profile import accept_five_updates
                        accept_five_updates(checkpoint, five_update_records)
                    else:
                        # A recovery from outer1..4 still owes the original
                        # fixed QA5. Its earlier updates belong to the source
                        # segment, so a same-process fresh-five receipt would
                        # be false; the history reader verifies both segments.
                        print('SEARCH_G1_RESUMED_FIXED_QA5_COMPLETE', flush=True)
                if (self.config.trainer.get('full_state_restore_receipt',None) and
                        not easy_restore and
                        self.global_steps==min(continuation_k+5,self.total_training_steps-1)):
                    from search_g1.full_state_restore import accept_resumed_five
                    accept_resumed_five(five_update_records)
                if self.global_steps in (50,100,150,200,250) and self.global_steps + 1 < self.total_training_steps:
                    from reproduction.profile import refresh_after_checkpoint
                    self.reward_fn.expected_probe_version = refresh_after_checkpoint(self.global_steps, checkpoint)

                if checkpoint is not None and self.qprobe_runtime is not None and selection_mode!='dynrpo7b':
                    from search_g1.qprobe_checkpoint import publish_completed_pair
                    publish_completed_pair(self, checkpoint)

                if checkpoint is not None and selection_mode=='dynrpo7b':
                    from reproduction.dynrpo7b.rolling import publish_boundary
                    publish_boundary(self,checkpoint)
                if checkpoint is not None and self.config.trainer.get('rolling_checkpoint', False):
                    # The full model/Adam snapshot is already on disk. Finish
                    # QA/refit first, then pin the small driver/worker RNG state
                    # for the next outer before replacing the previous pointer.
                    from search_g1.easy2560_restore_runtime import capture_boundary
                    from search_g1.easy2560_rolling_checkpoint import publish_checkpoint
                    boundary = capture_boundary(self, checkpoint)
                    rolling = publish_checkpoint(os.environ['SEARCH_G1_BASE'], checkpoint,
                                                 boundary_manifest=boundary)
                    print('SEARCH_G1_ROLLING_CHECKPOINT=' + json.dumps(rolling), flush=True)

                self.global_steps += 1

                if self.global_steps >= self.total_training_steps:

                    # perform validation after training
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate(final=True)
                        pprint(f'Final validation metrics: {val_metrics}')
                        print('SEARCH_G1_FINAL_METRICS_JSON=' + __import__('json').dumps(val_metrics,sort_keys=True),flush=True)
                        logger.log(data=val_metrics, step=self.global_steps - 1)
                    return

    def _create_loss_mask(self, batch, metrics):
        """Create loss mask for state tokens."""
        response_length = batch.batch['responses'].shape[-1]
        response_mask = batch.batch['attention_mask'][:, -response_length:]

        loss_mask = batch.batch['info_mask'][:, -response_length:]
        batch.batch['loss_mask'] = loss_mask

        metrics.update({
            'state_tokens/total': loss_mask.sum().item(),
            'state_tokens/coverage': (loss_mask.sum() / response_mask.sum()).item(),
        })

        return batch, metrics
