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
import uuid
import time
import functools
import itertools
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


def _baseline_validation(function):
    @functools.wraps(function)
    def wrapped(self, *args, **kwargs):
        controller = getattr(self, 'baseline_controller', None)
        if controller is None or controller.ranker is None:
            return function(self, *args, **kwargs)
        old = controller.ranker.validation
        controller.ranker.validation = True
        try:
            return function(self, *args, **kwargs)
        finally:
            controller.ranker.validation = old
    return wrapped


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


def cast_rollout_structural_fields(data, actor_config, preserve_old=False):
    """Keep exact legacy casts; the explicit mu profile preserves all floats."""
    from search_r1.baselines.policy_safety_runtime import behavior_mu_correction_enabled
    corrected = behavior_mu_correction_enabled(actor_config.get('policy_safety', {}))
    for key in data.batch.keys():
        if (preserve_old and key == 'old_log_probs') or (corrected and data.batch[key].is_floating_point()):
            continue
        data.batch[key] = data.batch[key].long()


def prepare_behavior_mu_old_policy(data, actor_config, proof_cache=None):
    from search_r1.baselines.policy_safety_runtime import behavior_mu_correction_enabled
    corrected = behavior_mu_correction_enabled(actor_config.get('policy_safety', {}))
    if not corrected:
        return False
    from search_r1.baselines.tree_behavior_mu import validate_behavior_mu_batch, runtime_provenance
    validate_behavior_mu_batch(data, expected_provenance=runtime_provenance(os.environ), required=True, proof_cache=proof_cache)
    from search_r1.baselines.policy_safety_runtime import full_actor_beta_available
    sampled = data.batch['sampled_old_log_probs']
    if not full_actor_beta_available(data.meta_info):
        if 'old_log_probs' in data.batch.keys() or 'actor_old_log_probs' in data.batch.keys():
            raise ValueError('Omitted full actor beta must not have a fabricated substitute')
        if not actor_config.use_kl_loss or actor_config.kl_loss_coef != 0.:
            raise ValueError('Omitted full beta requires reference KL loss zero')
        # Native behavior-mu validation above proves every sampled distribution
        # is plain temperature 1. The omitted HF path must not be a metadata producer.
        temperature = data.meta_info.get('temperature', 1.)
        if type(temperature) not in (int, float) or temperature != 1.:
            raise ValueError('Actor temperature differs from actual captured behavior-mu distribution')
        data.meta_info['temperature'] = 1.
        data.batch['old_log_probs'] = sampled.detach().clone()
        data.meta_info['tree_behavior_mu_corrected_profile'] = True
        return True
    beta = data.batch['old_log_probs']
    if not beta.is_floating_point() or beta.shape != sampled.shape or sampled.dtype != torch.float32:
        raise ValueError('Behavior-mu profile requires independent floating actor beta and FP32 sampled mu')
    data.batch['actor_old_log_probs'] = beta.detach().clone()
    data.batch['old_log_probs'] = sampled.detach().clone()
    data.meta_info['tree_behavior_mu_corrected_profile'] = True
    return True


def validate_final_behavior_mu_batch(data, actor_config, proof_cache=None):
    from search_r1.baselines.policy_safety_runtime import behavior_mu_correction_enabled
    if not behavior_mu_correction_enabled(actor_config.get('policy_safety', {})):
        return
    from search_r1.baselines.tree_behavior_mu import validate_behavior_mu_batch, runtime_provenance
    receipt = validate_behavior_mu_batch(data, expected_provenance=runtime_provenance(os.environ), required=True, proof_cache=proof_cache)
    from search_r1.baselines.policy_safety_runtime import full_actor_beta_available
    if (data.meta_info.get('tree_behavior_mu_corrected_profile') is not True
            or full_actor_beta_available(data.meta_info) != ('actor_old_log_probs' in data.batch.keys())
            or not torch.equal(data.batch['old_log_probs'], data.batch['sampled_old_log_probs'])):
        raise ValueError('Final balanced behavior-mu batch has no fixed mu/independent beta binding')
    return receipt


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch['info_mask'] if 'info_mask' in data.batch else data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    if 'ref_log_prob' in data.batch.keys():
        # The behavior-mu PPO denominator does not change the original beta reward KL.
        reward_policy_log_probs = (data.batch['actor_old_log_probs']
                                   if data.meta_info.get('tree_behavior_mu_corrected_profile') is True
                                   else data.batch['old_log_probs'])
        kld = core_algos.kl_penalty(torch.where(response_mask.bool(), reward_policy_log_probs, 0.),
                                    torch.where(response_mask.bool(), data.batch['ref_log_prob'], 0.),
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


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1):
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
                                                                        index=index)
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


@contextmanager
def _driver_cpu_stage(name, outer):
    start=time.perf_counter();cpu=time.process_time();successful=False
    print('TREE_DRIVER_STAGE='+json.dumps(dict(event='start',stage=name,outer=outer,
        unix_time=time.time(),pid=os.getpid())),flush=True)
    try:
        yield
        successful=True
    finally:
        print('TREE_DRIVER_STAGE='+json.dumps(dict(event='complete',stage=name,outer=outer,
            unix_time=time.time(),pid=os.getpid(),successful=successful,wall_seconds=time.perf_counter()-start,
            cpu_seconds=time.process_time()-cpu)),flush=True)


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
        self.baseline_controller = None
        if self.config.get('baselines') is not None and self.config.baselines.method != 'grpo':
            from search_r1.baselines.controller import Controller, validate_config
            validate_config(self.config)
            self.baseline_controller = Controller(self)
    
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

        if self.config.trainer.get('training_state_enabled', False):
            from search_r1.baselines.training_state import CheckpointBatchSampler
            self.checkpoint_sampler = CheckpointBatchSampler(
                len(self.train_dataset), self.config.data.train_batch_size, self.config.baselines.seed,
                os.environ['SEARCH_R1_BASELINE_DATA_IDENTITY'], self.config.data.shuffle_train_dataloader)
            self.train_dataloader = DataLoader(dataset=self.train_dataset, batch_sampler=self.checkpoint_sampler,
                                               num_workers=0, collate_fn=collate_fn)
        else:
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

        if self.config.get('baselines') is not None and self.config.baselines.holdout.get('input_mode') == 'prebuilt_manifest':
            from search_r1.baselines.holdout import _validate_prebuilt, frame_identity, ordered_question_ids_hash
            split = _validate_prebuilt(self.config)
            if split['content_identity']['sha256'] != os.environ.get('SEARCH_R1_BASELINE_DATA_IDENTITY'):
                raise ValueError('Worker prebuilt data identity differs from entry')
            for name, dataset in [('train', self.train_dataset), ('validation', self.val_dataset)]:
                record = split['manifest']['splits'][name]
                identity = frame_identity(dataset.dataframe, self.config.data.prompt_key)
                if (len(dataset) != record['rows'] or identity != record['content_identity']
                        or ordered_question_ids_hash(dataset.dataframe) != record['question_ids_sha256']):
                    raise ValueError('Worker dataset was filtered, reordered or changed: ' + name)

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
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            actual_limit = self.config.trainer.get('actual_update_limit', None)
            if actual_limit is not None and int(actual_limit)+1 != total_training_steps:
                raise ValueError('Actual update limit differs from legacy terminal counter')
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps if actual_limit is None else int(actual_limit)
            self.config.critic.optim.total_training_steps = total_training_steps

    def _prepare_final_validation(self):
        """Switch only terminal evaluation to the complete held-out data."""
        files=self.config.data.get('final_val_files')
        if not files:
            return
        from torch.utils.data import DataLoader
        from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
        self.val_dataset=RLHFDataset(
            parquet_files=files, tokenizer=self.tokenizer,
            prompt_key=self.config.data.prompt_key,
            max_prompt_length=self.config.data.max_prompt_length,
            filter_prompts=True,
            return_raw_chat=self.config.data.get('return_raw_chat',False),
            truncation='error')
        self.val_dataloader=DataLoader(
            dataset=self.val_dataset, batch_size=self.config.data.val_batch_size,
            shuffle=False, drop_last=False, collate_fn=collate_fn)
        print(f'Final validation dataset size: {len(self.val_dataset)}',flush=True)

    @_baseline_validation
    def _validate(self):
        """
        The training loop of PPO with global metric computation.
        Accumulates metrics across all batches before computing final statistics.
        """
        import torch
        reward_tensor_lst = []
        data_source_lst = []
        qa_rows = []

        def collect_qa(batch):
            if not self.config.get('baselines'):
                return
            from search_r1.baselines.evaluation import batch_answer_metrics
            qa_rows.extend(batch_answer_metrics(batch, self.tokenizer))

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
            force_final_answer=self.config.get('force_final_answer', True),
        )

        # Agent config preparation
        generation_manager = LLMGenerationManager(
            tokenizer=self.tokenizer,
            actor_rollout_wg=self.actor_rollout_wg,
            config=gen_config,
            is_validation = True,
        )
        controller = getattr(self, 'baseline_controller', None)
        if controller is not None and controller.ranker is not None:
            from search_r1.baselines.controller import ranked_validation_manager
            generation_manager = ranked_validation_manager(generation_manager, controller)
        elif (controller is not None and controller.method == 'tree_grpo' and
              controller.config.tree.get('token_preservation',{}).get('enabled',False)):
            from search_r1.baselines.controller import token_preserving_validation_manager
            generation_manager = token_preserving_validation_manager(generation_manager, controller)

        if not self.config.do_search:
            for test_data in self.val_dataloader:
                test_batch = DataProto.from_single_dict(test_data)

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

                collect_qa(test_batch)
                reward_tensor_lst.append(reward_tensor)
                data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))
        else:
            for batch_dict in self.val_dataloader:
                timing_raw = {}
                test_batch: DataProto = DataProto.from_single_dict(batch_dict)
                # test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n_agent, interleave=True)
                
                test_gen_batch = test_batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
                test_gen_batch.meta_info = {
                    'eos_token_id': self.tokenizer.eos_token_id,
                    'pad_token_id': self.tokenizer.pad_token_id,
                    'recompute_log_prob': False,
                    'do_sample': False,
                    'validate': True,
                }
                with _timer('step', timing_raw):
                    first_input_ids = test_gen_batch.batch['input_ids'][:, -gen_config.max_start_length:].clone()
                    with _timer('gen', timing_raw):
                        generation_manager.timing_raw = timing_raw
                        final_gen_batch_output = generation_manager.run_llm_loop(
                            gen_batch=test_gen_batch,
                            initial_input_ids=first_input_ids,
                        )
                    
                    test_batch = test_batch.union(final_gen_batch_output)
                    
                    for key in test_batch.batch.keys():
                        test_batch.batch[key] = test_batch.batch[key].long()
                    
                    # evaluate using reward_function
                    # for certain reward function (e.g. sandbox), the generation can overlap with reward
                    reward_tensor = self.val_reward_fn(test_batch)

                    collect_qa(test_batch)
                    reward_tensor_lst.append(reward_tensor)
                    data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))

        reward_tensor = torch.cat([rw.sum(-1) for rw in reward_tensor_lst], dim=0).cpu()  # (batch_size,)
        # reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)
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

        if qa_rows:
            from search_r1.baselines.evaluation import aggregate_answer_metrics
            metric_dict.update(aggregate_answer_metrics(qa_rows))
        if self.config.get('baselines'):
            print('BASELINE_EVALUATION_METRICS_JSON=' + json.dumps(metric_dict, sort_keys=True), flush=True)
        from search_r1.baselines.retriever_phase import release_retriever_gpu
        release_retriever_gpu(self.config,'after_validation')
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

    def _evaluate_with_checkpoint_protection(self, save_due):
        """Keypoint state exists before any evaluator can fail; never overwrite it."""
        if save_due:
            self._save_checkpoint()
        metrics=self._validate()
        if save_due and self.config.trainer.get('training_state_enabled',False):
            # Evaluation can advance driver/worker RNG. Keep its actual state in
            # a separate immutable snapshot instead of calling it pre-eval state.
            self._save_complete_training_state(boundary_phase='post-evaluation')
        return metrics

    def _save_checkpoint(self):
        actor_local_path = os.path.join(self.config.trainer.default_local_dir, 'actor',
                                        f'global_step_{self.global_steps}')
        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, 'actor')
        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path)

        if self.use_critic:
            critic_local_path = os.path.join(self.config.trainer.default_local_dir, 'critic',
                                             f'global_step_{self.global_steps}')
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, 'critic')
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path)
        if getattr(self, 'baseline_controller', None) is not None:
            self.baseline_controller.save(self.global_steps)
        if self.config.trainer.get('training_state_enabled', False):
            self._save_complete_training_state()

    def _save_complete_training_state(self, boundary_phase=None):
        from search_r1.baselines.training_state import atomic_save, driver_snapshot, restore_driver, digest
        identity = dict(job_id=__import__('search_r1.baselines.execution_identity',fromlist=['execution_id']).execution_id(), code_commit=os.environ['SEARCH_R1_BASELINE_COMMIT'],
                        config_sha256=os.environ['SEARCH_R1_BASELINE_CONFIG_SHA256'],
                        data_identity=os.environ['SEARCH_R1_BASELINE_DATA_IDENTITY'],
                        model_identity=os.environ['SEARCH_R1_MODEL_IDENTITY'], completed_step=self.global_steps,
                        actual_update_limit=self.config.trainer.actual_update_limit)
        if self.actual_outer_updates != self.global_steps:
            raise ValueError('Saved actual update counter differs from completed step')
        from search_r1.baselines.tree_guard_continue import enabled as continue_enabled, accounting
        if continue_enabled(self.config.actor_rollout_ref.actor):
            identity['guard_continue_accounting'] = accounting(self)
        directory = os.path.join(self.config.trainer.default_local_dir, 'training_state', f'global_step_{self.global_steps}')
        if boundary_phase is not None:
            if boundary_phase != 'post-evaluation':raise ValueError('Unknown checkpoint boundary phase')
            directory=os.path.join(directory,boundary_phase)
        verify = boundary_phase is None and self.global_steps == self.config.trainer.acceptance_step
        workers = self.actor_rollout_wg.save_training_state(directory, identity, verify=verify)
        if len(workers) != self.actor_rollout_wg.world_size or {r['identity']['rank'] for r in workers} != set(range(len(workers))):
            raise ValueError('Incomplete worker snapshot coverage')
        if verify and not all(r['exact_same_world_reload_verified'] for r in workers):
            raise ValueError('Worker reload verification missing')
        saved = driver_snapshot(self, identity); logical = digest(saved)
        driver_path = os.path.join(directory, 'driver.pt'); driver_sha = atomic_save(saved, driver_path)
        if verify:
            import torch
            loaded = torch.load(driver_path, map_location='cpu', weights_only=False)
            if digest(loaded) != logical: raise ValueError('Driver serialized state differs')
            restore_driver(self, loaded, identity)
            if digest(driver_snapshot(self, identity)) != logical: raise ValueError('Driver reloaded state differs')
        from search_r1.baselines.select_teacher import model_manifest
        hf_path = os.path.join(self.config.trainer.default_local_dir, 'actor', f'global_step_{self.global_steps}')
        hf_identity = model_manifest(hf_path)
        manifest = dict(status='COMPLETE_SAME_WORLD_STATE_SAVED', identity=identity, worker_states=workers,
                        driver=dict(path=driver_path, sha256=driver_sha, logical_state_sha256=logical,
                                    next_step=self.global_steps+1, exact_reload_verified=verify),
                        hf_checkpoint=hf_path, hf_identity=hf_identity,
                        limits='Same-world boundary snapshot and live continuation; no arbitrary executor/vLLM restart equivalence')
        if boundary_phase is not None:
            manifest['boundary_phase']=boundary_phase
            manifest['limits']+='; captures actual completed scheduled evaluation RNG, not a pre-evaluation state'
        manifest_path = os.path.join(directory, 'manifest.json')
        manifest_sha = atomic_save(manifest, manifest_path, json_format=True)
        if verify:
            self.acceptance_ready = dict(identity, completed_outer_updates=self.global_steps,
                                         next_outer_update=self.global_steps+1, full_state_manifest=manifest_path,
                                         full_state_manifest_sha256=manifest_sha, full_state_reload_verified=True,
                                         hf_checkpoint=hf_path, hf_identity=hf_identity)

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

        logger = self.logger
        self.global_steps = 0
        from search_r1.baselines.tree_fresh200 import is_fresh200
        fresh200 = self.config.get('baselines') is not None and is_fresh200(self.config)
        from search_r1.baselines.tree_current_chain_resume import enabled as current_chain_enabled
        current_chain_resume = fresh200 and current_chain_enabled()
        if fresh200:
            from search_r1.baselines.tree_fresh200 import (validate_fresh200_config, refuse_recovery_environment,
                                                          validate_fresh_sampler)
            from search_r1.baselines.training_state import atomic_save
            validate_fresh200_config(self.config)
            refuse_recovery_environment()
            validate_fresh_sampler(self.checkpoint_sampler)
            if getattr(self, 'actual_outer_updates', 0) != 0:
                raise RuntimeError('Fresh200 driver is not in a new initialization state')
            receipts = self.actor_rollout_wg.inspect_fresh_initialization(self.config.actor_rollout_ref.model.path)
            if (len(receipts) != 8 or {r['rank'] for r in receipts} != set(range(8))
                    or any(r['optimizer_state_entries'] != 0 or r['scheduler_step'] != 0 for r in receipts)):
                raise RuntimeError('Fresh200 actual worker initialization proof incomplete')
            self.actual_outer_updates = 0
            self.fresh_initialization_provenance = dict(status=(
                'COLD_OFFICIAL_INITIAL_STATE_OBSERVED_BEFORE_CURRENT_CHAIN_MIGRATION'
                if current_chain_resume else 'FRESH_OFFICIAL_INITIAL_STATE_OBSERVED'),
                calibration_profile='fresh_base_zero_v1', training_protocol='tree_fresh200_five_leaf_v1',
                initialized_global_steps=self.global_steps, driver_completed_outer=self.actual_outer_updates,
                parent_completed_step=self.actual_outer_updates, parent_cursor=self.checkpoint_sampler.cursor,
                checkpoint_loaded=False, workers=receipts,
                code_commit=os.environ['SEARCH_R1_BASELINE_COMMIT'],
                config_sha256=os.environ['SEARCH_R1_BASELINE_CONFIG_SHA256'],
                data_identity=os.environ['SEARCH_R1_BASELINE_DATA_IDENTITY'],
                model_identity=os.environ['SEARCH_R1_MODEL_IDENTITY'],
                job_id=__import__('search_r1.baselines.execution_identity',fromlist=['execution_id']).execution_id(), run_id=os.environ['SEARCH_R1_BASELINE_RUN_ID'])
            atomic_save(self.fresh_initialization_provenance,
                os.path.join(os.environ['SEARCH_R1_BASELINE_OUTPUT'], 'fresh-initialization.json'), json_format=True)
            if current_chain_resume:
                from search_r1.baselines.tree_current_chain_resume import migrate_trainer
                migrate_trainer(self)
        if os.environ.get('TREE_RECOVERY_MANIFEST'):
            from search_r1.baselines.training_state import recovery_manifest,restore_recovery_driver,atomic_save
            manifest=recovery_manifest()
            worker_receipts=self.actor_rollout_wg.load_recovery_training_state()
            if len(worker_receipts)!=8 or {r['rank'] for r in worker_receipts}!=set(range(8)):
                raise ValueError('Cold recovery rank receipt incomplete')
            if any(r['scheduler']!=manifest['identity']['completed_step'] for r in worker_receipts):
                raise ValueError('Cold recovery scheduler boundary mismatch')
            migration=restore_recovery_driver(self,manifest)
            atomic_save(dict(status='COLD_PARENT_STATE_LOADED_PENDING_EVALUATION',workers=worker_receipts,
                sampler_migration=migration,provenance=self.recovery_provenance),
                os.path.join(os.environ['SEARCH_R1_BASELINE_OUTPUT'],'recovery-loaded.json'),json_format=True)
        # perform validation before training
        # currently, we only support validation using the reward_function.
        if (not current_chain_resume and self.val_reward_fn is not None
                and self.config.trainer.get('val_before_train', True)):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if os.environ.get('TREE_RECOVERY_MANIFEST'):
                if val_metrics.get('val/f1/all',0)<self.config.trainer.recovery_min_f1:
                    raise RuntimeError('Restored fixed512 evaluation below recovery quality gate')
            if self.config.trainer.get('val_only', False):
                return val_metrics

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
            force_final_answer=self.config.get('force_final_answer', True),
        )

        generation_manager = LLMGenerationManager(
            tokenizer=self.tokenizer,
            actor_rollout_wg=self.actor_rollout_wg,
            config=gen_config,
        )

        # start training loop
        # Refused attempts consume real data; completed-outer target, rather
        # than a fixed number of attempts/epochs, terminates fresh training.
        for epoch in (itertools.count() if fresh200 else range(self.config.trainer.total_epochs)):
            batch_iterator = iter(self.train_dataloader)
            while True:
                outer_mu_proofs=None
                # Wait BEFORE fetching the sixth question batch. Keep every
                # actor, Adam, scheduler, Tree RNG and sampler live in place.
                if (self.config.trainer.get('acceptance_gate_enabled', False)
                        and self.global_steps == self.config.trainer.acceptance_step+1
                        and not getattr(self, 'acceptance_released', False)):
                    from search_r1.baselines.training_state import wait_for_acceptance
                    wait_for_acceptance(os.path.join(os.environ['SEARCH_R1_BASELINE_OUTPUT'],
                                                    f'acceptance-step{self.config.trainer.acceptance_step}'),
                                        self.acceptance_ready, self.config.trainer.acceptance_gate_timeout_seconds)
                    self.acceptance_released = True
                try:
                    batch_dict = next(batch_iterator)
                except StopIteration:
                    break
                print(f'epoch {epoch}, step {self.global_steps}')
                metrics = {}
                timing_raw = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)
                if fresh200:
                    from search_r1.baselines.tree_guard_continue import begin_attempt
                    attempt_metadata = begin_attempt(self)
                    batch.meta_info.update(attempt_metadata)
                batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n_agent, interleave=True)

                # pop those keys for generation
                gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])

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
                        first_input_ids = gen_batch.batch['input_ids'][:, -gen_config.max_start_length:].clone().long()

                        with _timer('gen', timing_raw):
                            generation_manager.timing_raw = timing_raw
                            if self.baseline_controller is None:
                                final_gen_batch_output = generation_manager.run_llm_loop(
                                    gen_batch=gen_batch,
                                    initial_input_ids=first_input_ids,
                                )
                            else:
                                final_gen_batch_output, sources = self.baseline_controller.generate(
                                    generation_manager, gen_batch, batch, metrics)
                                from search_r1.baselines.rollout import select_rows
                                batch = select_rows(batch, sources)

                        # final_gen_batch_output.batch.apply(lambda x: x.long(), inplace=True)
                        cast_rollout_structural_fields(final_gen_batch_output, self.config.actor_rollout_ref.actor)

                        if os.environ.get('RECURSIVE_A_OMIT_FULL_BETA') == '1':
                            actor_config = self.config.actor_rollout_ref.actor
                            from search_r1.baselines.policy_safety_runtime import behavior_mu_correction_enabled
                            from search_r1.baselines.tree_guard_continue import execution_profile_bound
                            if (not execution_profile_bound()
                                    or not behavior_mu_correction_enabled(actor_config.policy_safety)
                                    or not actor_config.use_kl_loss or actor_config.kl_loss_coef != 0.
                                    or self.config.algorithm.kl_ctrl.kl_coef != 0.):
                                raise ValueError('No-full-beta profile exceeds authorized zero-KL behavior-mu scope')
                            self.actor_rollout_wg.release_rollout_policy_cache()
                            from search_r1.baselines.retriever_phase import release_retriever_gpu
                            release_retriever_gpu(self.config,'after_tree_sampling_before_hf')
                            final_gen_batch_output.meta_info['tree_full_actor_beta_available'] = False
                        else:
                            with torch.no_grad():
                                output = self.actor_rollout_wg.compute_log_prob(final_gen_batch_output)
                                final_gen_batch_output = final_gen_batch_output.union(output)
                        from search_r1.baselines.tree_behavior_mu import BehaviorMuProofCache
                        outer_mu_proofs=BehaviorMuProofCache()
                        with _driver_cpu_stage('prepare_behavior_mu',self.global_steps):
                            prepare_behavior_mu_old_policy(final_gen_batch_output, self.config.actor_rollout_ref.actor,outer_mu_proofs)

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
                    with _driver_cpu_stage('balance',self.global_steps):
                        self._balance_batch(batch, metrics=metrics)
                    with _driver_cpu_stage('balanced_mu_validate',self.global_steps):
                        validate_final_behavior_mu_batch(batch, self.config.actor_rollout_ref.actor,outer_mu_proofs)

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    # batch.batch.apply(lambda x, key: x.long() if key != "old_log_probs" else x, inplace=True, key=True)
                    cast_rollout_structural_fields(batch, self.config.actor_rollout_ref.actor, preserve_old=True)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer('ref', timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer('adv', timing_raw), _driver_cpu_stage('reward_advantages',self.global_steps):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_tensor = self.reward_fn(batch)
                        batch.batch['token_level_scores'] = reward_tensor

                        # compute rewards. apply_kl_penalty if available
                        if self.baseline_controller is not None and self.baseline_controller.method in ('topd', 'opsd'):
                            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']
                        elif not self.config.actor_rollout_ref.actor.use_kl_loss:
                            batch, kl_metrics = apply_kl_penalty(batch,
                                                                 kl_ctrl=self.kl_ctrl,
                                                                 kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        # compute advantages, executed on the driver process
                        if self.baseline_controller is None:
                            batch = compute_advantage(batch,
                                                      adv_estimator=self.config.algorithm.adv_estimator,
                                                      gamma=self.config.algorithm.gamma,
                                                      lam=self.config.algorithm.lam,
                                                      num_repeat=self.config.actor_rollout_ref.rollout.n)
                        else:
                            batch = self.baseline_controller.compute_advantages(batch, metrics)

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
                            if fresh200:
                                batch.meta_info.update(attempt_metadata)
                            with _driver_cpu_stage('pre_actor_mu_validate',self.global_steps):
                                validate_final_behavior_mu_batch(batch, self.config.actor_rollout_ref.actor,outer_mu_proofs)
                            audit_controller = getattr(self, 'baseline_controller', None)
                            if hasattr(audit_controller, 'audit_actor_batch'):
                                with _driver_cpu_stage('actor_batch_audit',self.global_steps):
                                    audit_controller.audit_actor_batch(batch)
                            from search_r1.baselines.recording_schedule import record_details
                            if os.environ.get('SEARCH_R1_TRAJECTORY_DIR') and record_details(self.global_steps):
                                from search_r1.baselines.trajectory_log import save_rollouts
                                with _driver_cpu_stage('save_rollouts',self.global_steps):
                                    save_rollouts(batch,self.tokenizer,self.global_steps,
                                                  os.environ['SEARCH_R1_TRAJECTORY_DIR'])
                            if os.environ.get('TREE_RECOVERY_MANIFEST'):
                                reward=float(batch.batch['token_level_rewards'].sum(-1).mean())
                                if reward<self.config.trainer.recovery_min_train_reward:
                                    raise RuntimeError(f'Recovery reward gate: {reward}; rollout evidence saved before update')
                            from search_r1.baselines.runtime_audit import enabled as audit_enabled
                            if audit_enabled():
                                batch.meta_info['baseline_outer_step'] = self.global_steps
                                from search_r1.baselines.runtime_audit import driver_outer
                                driver_outer(batch, self.global_steps, self.config.baselines.method)
                            if os.environ.get('TREE_PPO_CALIBRATION_ONLY') == '1':
                                from search_r1.baselines.sampling_consistency import finalize_calibration
                                return finalize_calibration(self, batch)
                            if fresh200:
                                from search_r1.baselines.tree_partial_state import partial_context
                                batch.meta_info['tree_partial_context'] = partial_context(self)
                            try:
                                actor_output = self.actor_rollout_wg.update_actor(batch)
                            except Exception as actor_error:
                                if fresh200:
                                    from search_r1.baselines.tree_partial_state import preserve_unmatched_actor_failure
                                    try:
                                        evidence = preserve_unmatched_actor_failure(self, batch, actor_error)
                                        print('TREE_UNMATCHED_ACTOR_FAILURE_EVIDENCE=' + evidence, flush=True)
                                    except Exception as evidence_error:
                                        print('TREE_UNMATCHED_ACTOR_FAILURE_EVIDENCE_FAILED=' +
                                              repr(evidence_error), flush=True)
                                raise
                            if fresh200:
                                from search_r1.baselines.tree_partial_state import stop_on_partial_actor_result
                                if stop_on_partial_actor_result(self, batch, actor_output):
                                    metrics.update({'training/attempt_id': self.training_attempts,
                                        'training/refused_attempts': self.refused_attempts,
                                        'training/actual_outer_updates': self.actual_outer_updates})
                                    logger.log(data=metrics, step=self.training_attempts)
                                    continue
                                from search_r1.baselines.tree_guard_continue import validate_completed_result
                                validate_completed_result(self, actor_output)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)
                        if getattr(self, 'baseline_controller', None) is not None:
                            self.baseline_controller.after_actor(batch, metrics)

                    baseline_settings = getattr(self.config, 'baselines', None)
                    if baseline_settings is not None and baseline_settings.method in ('grpo', 'gigpo', 'tree_grpo'):
                        from search_r1.baselines.tree_guard_continue import round_progress_enabled
                        zero_write_round = (fresh200 and round_progress_enabled(self.config.actor_rollout_ref.actor)
                            and metrics.get('actor/guard_round_progress_enabled') == 1
                            and metrics.get('actor/zero_write_sampling_round') == 1
                            and metrics.get('actor/optimizer_opportunities_processed') == 20
                            and metrics.get('actor/optimizer_steps') == 0)
                        if metrics.get('actor/optimizer_steps', 0) <= 0 and not zero_write_round:
                            raise RuntimeError('Outer iteration performed no actor update; budget cannot advance')
                        if fresh200:
                            from search_r1.baselines.tree_fresh200 import validate_completed_outer
                            validate_completed_outer(metrics, self.global_steps)
                            from search_r1.baselines.tree_guard_continue import record_completed_result
                            record_completed_result(self, actor_output)
                        self.actual_outer_updates = getattr(self, 'actual_outer_updates', 0) + 1
                        metrics['training/actual_outer_updates'] = self.actual_outer_updates
                        if fresh200:
                            metrics['training/attempt_id'] = self.training_attempts
                            metrics['training/refused_attempts'] = getattr(self, 'refused_attempts', 0)
                            if round_progress_enabled(self.config.actor_rollout_ref.actor):
                                metrics['training/consumed_sampling_rounds'] = self.actual_outer_updates
                                metrics['training/zero_write_sampling_rounds'] = self.zero_write_sampling_rounds
                                metrics['training/optimizer_write_rounds'] = self.actual_outer_updates-self.zero_write_sampling_rounds
                                metrics['training/normal_guard_rounds'] = self.normal_guard_rounds
                                metrics['training/actual_adam_calls_per_rank'] = actor_output.meta_info[
                                    'tree_fresh_actor_result']['ranks'][0]['total_optimizer_calls']
                        if os.environ.get('TREE_RECOVERY_MANIFEST') and metrics['actor/optimizer_steps']!=10:
                            raise RuntimeError('Recovery outer optimizer calls differ from required10')

                    recursive_rolling = (self.config.get('baselines') is not None and
                        self.config.baselines.tree.get('branch_sampling') == 'recursive_uniform_v1')
                    explicit_save_steps = self.config.trainer.get('checkpoint_save_steps', None)
                    save_due = (self.global_steps in explicit_save_steps if explicit_save_steps is not None
                                else self.config.trainer.save_freq > 0 and
                            (recursive_rolling or self.global_steps % self.config.trainer.save_freq == 0 or
                             self.global_steps == getattr(self.config.trainer, 'acceptance_step', 0) or
                             self.global_steps + 1 >= self.total_training_steps))
                    # validate
                    explicit_eval_steps = self.config.trainer.get('evaluation_steps', None)
                    evaluation_due = (self.global_steps in explicit_eval_steps if explicit_eval_steps is not None
                                      else self.config.trainer.test_freq > 0 and
                                      self.global_steps % self.config.trainer.test_freq == 0)
                    protected_eval_checkpoint=(self.val_reward_fn is not None and evaluation_due and save_due)
                    if self.val_reward_fn is not None and evaluation_due:
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._evaluate_with_checkpoint_protection(save_due)
                        metrics.update(val_metrics)
                        if os.environ.get('TREE_RECOVERY_MANIFEST'):
                            if val_metrics.get('val/f1/all',0)<self.config.trainer.recovery_min_f1:
                                logger.log(data=val_metrics,step=self.global_steps)
                                raise RuntimeError('Recovery fixed512 evaluation quality gate')
                    if (getattr(self, 'baseline_controller', None) is not None
                            and metrics.get('actor/optimizer_steps', 0) > 0):
                        with _timer('stage_testing', timing_raw):
                            metrics.update(self.baseline_controller.stage_validation(self.global_steps))

                    if save_due:
                        # Save the final updated policy before the counter advances.
                        # Final validation below must have a matching saved model,
                        # even when the last update is not a save-frequency boundary.
                        with _timer('save_checkpoint', timing_raw):
                            if not protected_eval_checkpoint:
                                self._save_checkpoint()
                            if recursive_rolling and explicit_save_steps is None:
                                from search_r1.baselines.recursive_rolling import prune
                                prune(self.config.trainer.default_local_dir, self.global_steps,
                                      os.environ['SEARCH_R1_BASELINE_CONFIG_SHA256'],
                                      os.environ['SEARCH_R1_BASELINE_COMMIT'])

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                self.global_steps += 1

                if self.global_steps >= self.total_training_steps:

                    # perform validation after training
                    if self.val_reward_fn is not None and not getattr(self.config.trainer, 'smoke_only', False):
                        self._prepare_final_validation()
                        val_metrics = self._validate()
                        pprint(f'Final validation metrics: {val_metrics}')
                        print('SEARCH_R1_FINAL_METRICS_JSON=' + json.dumps(val_metrics, sort_keys=True), flush=True)
                        logger.log(data=val_metrics, step=self.global_steps)
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
