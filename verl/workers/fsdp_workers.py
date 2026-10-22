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
The main entry point to run the PPO algorithm
"""

import logging
import os
import warnings

import torch
import torch.distributed
import verl.utils.hdfs_io as hdfs_io
import verl.utils.torch_functional as verl_F
from omegaconf import DictConfig, open_dict
from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import register, Dispatch
from verl.utils import hf_tokenizer
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.fsdp_utils import get_fsdp_wrap_policy, offload_fsdp_grad, init_fn, get_init_weight_context_manager
from verl.utils.fsdp_utils import offload_fsdp_optimizer, offload_fsdp_param_and_grad, load_fsdp_optimizer, \
    load_fsdp_param_and_grad
from verl.utils.import_utils import import_external_libs
from verl.utils.model import compute_position_id_with_mask
from verl.utils.flops_counter import FlopsCounter
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager

from codetiming import Timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_PPO_LOGGING_LEVEL', 'WARN'))


class ActorRolloutRefWorker(Worker):
    """
    This worker can be instantiated as a standalone actor or a standalone rollout or a standalone reference policy
    or a hybrid engine based on the config.rollout
    """

    def __init__(self, config: DictConfig, role: str):
        super().__init__()
        from search_g1.process_jit_cache import configure
        self.process_jit_cache = configure()
        if os.environ.get('SEARCH_G1_PROFILE') == 'common8':
            from reproduction.common8.batch_contract import independent_worker_config
            config = independent_worker_config(config)
        self.config = config
        import torch.distributed
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")

        # build device mesh for FSDP
        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh
        # TODO(sgm): support FSDP hybrid shard for larger model
        self.device_mesh = init_device_mesh('cuda', mesh_shape=(world_size,), mesh_dim_names=['fsdp'])

        # build device mesh for Ulysses Sequence Parallel
        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.actor.get('ulysses_sequence_parallel_size', 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh('cuda',
                                                        mesh_shape=(dp, self.ulysses_sequence_parallel_size),
                                                        mesh_dim_names=['dp', 'sp'])

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        self.role = role
        assert self.role in ['actor', 'rollout', 'ref', 'actor_rollout', 'actor_rollout_ref']

        self._is_actor = self.role in ['actor', 'actor_rollout', 'actor_rollout_ref']
        self._is_rollout = self.role in ['rollout', 'actor_rollout', 'actor_rollout_ref']
        self._is_ref = self.role in ['ref', 'actor_rollout_ref']

        self._is_offload_param = False
        self._is_offload_grad = False
        self._is_offload_optimizer = False
        if self._is_actor:
            self._is_offload_param = self.config.actor.fsdp_config.get('param_offload', False)
            self._is_offload_grad = self.config.actor.fsdp_config.get('grad_offload', False)
            self._is_offload_optimizer = self.config.actor.fsdp_config.get('optimizer_offload', False)
        elif self._is_ref:
            # TODO: it seems that manual offload is slowly than FSDP offload
            self._is_offload_param = self.config.ref.fsdp_config.get('param_offload', False)

        # Check the supplied global values before in-place rank normalization.
        if os.environ.get('SEARCH_G1_PROFILE') == 'common8':
            from reproduction.common8.batch_contract import check_global
            check_global(self.config, self.device_mesh.shape[0],
                         self.ulysses_sequence_parallel_size)
        # normalize config
        if self._is_actor:
            self.config.actor.ppo_mini_batch_size //= (self.device_mesh.shape[0] // self.ulysses_sequence_parallel_size)
            self.config.actor.ppo_micro_batch_size //= (self.device_mesh.shape[0] //
                                                        self.ulysses_sequence_parallel_size)
            self.config.actor.ppo_mini_batch_size *= self.config.rollout.n
            self.config.actor.ppo_micro_batch_size *= self.config.rollout.n
        if self._is_rollout:
            self.config.rollout.log_prob_micro_batch_size //= (self.device_mesh.shape[0] //
                                                               self.ulysses_sequence_parallel_size)
            self.config.rollout.log_prob_micro_batch_size *= self.config.rollout.n
        if self._is_ref:
            self.config.ref.log_prob_micro_batch_size //= (self.device_mesh.shape[0] //
                                                           self.ulysses_sequence_parallel_size)
            self.config.ref.log_prob_micro_batch_size *= self.config.rollout.n

        if os.environ.get('SEARCH_G1_PROFILE') == 'common8':
            from reproduction.common8.batch_contract import check_local
            check_local(self.config, actor=self._is_actor,
                        rollout=self._is_rollout, reference=self._is_ref)

    def _build_model_optimizer(self,
                               model_path,
                               fsdp_config,
                               optim_config,
                               override_model_config,
                               use_remove_padding=False,
                               enable_gradient_checkpointing=False,
                               trust_remote_code=False):
        from verl.utils.model import print_model_size, update_model_config
        from verl.utils.torch_dtypes import PrecisionType
        from transformers import AutoModelForCausalLM, AutoConfig
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, MixedPrecision
        from torch import optim

        log_gpu_memory_usage('Before init from HF AutoModel', logger=logger)
        local_path = copy_local_path_from_hdfs(model_path)

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        # TODO(zhangchi.usc1992): 1. support create from random initialized model. 2. Support init with FSDP directly
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

        torch_dtype = fsdp_config.get('model_dtype', None)
        if torch_dtype is None:
            torch_dtype = torch.float32 if self._is_actor else torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(torch_dtype)

        # override model kwargs
        actor_model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)

        if use_remove_padding:
            from verl.models.registry import check_model_support_rmpad
            check_model_support_rmpad(actor_model_config.model_type)

        if use_remove_padding and self.ulysses_sequence_parallel_size > 1:
            from verl.models.transformers.monkey_patch import apply_monkey_patch
            apply_monkey_patch(actor_model_config, verbose=True)

        override_config_kwargs = {
            'bos_token_id': self.tokenizer.bos_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'pad_token_id': self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_model_config)
        update_model_config(actor_model_config, override_config_kwargs=override_config_kwargs)
        if self.rank == 0:
            print(f'Model config after override: {actor_model_config}')

        # NOTE(fix me): tie_word_embedding causes meta_tensor init to hang
        init_context = get_init_weight_context_manager(use_meta_tensor=not actor_model_config.tie_word_embeddings)

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            actor_module = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=local_path,
                                                                torch_dtype=torch_dtype,
                                                                config=actor_model_config,
                                                                attn_implementation=('flash_attention_2' if use_remove_padding else 'sdpa'),
                                                                trust_remote_code=trust_remote_code)
            # some parameters may not in torch_dtype. TODO(zhangchi.usc1992) remove this after we switch to fsdp2
            from verl.utils.rope_precision import cast_preserving_qwen_rope
            cast_preserving_qwen_rope(actor_module, torch_dtype)

            if enable_gradient_checkpointing:
                actor_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            if self._is_actor:
                from search_g1.easy7b_infra import flags as infra_flags
                if infra_flags().get('flash_attention_3_actor',False):
                    if not use_remove_padding or self.ulysses_sequence_parallel_size != 1:
                        raise ValueError('FA3 actor requires native remove-padding TP1/SP1 execution')
                    from search_g1.flash3_attention import install_flash3_actor
                    install_flash3_actor(actor_module,
                        compact_gqa=infra_flags().get('flash_attention_3_compact_gqa', False))
                if infra_flags()['selected_policy_logits']:
                    if not use_remove_padding or self.ulysses_sequence_parallel_size != 1:
                        raise ValueError('Selected policy projection requires native packed TP1/SP1 actor')
                    from search_g1.selected_qwen_head import install_selected_qwen_head
                    install_selected_qwen_head(actor_module)
        torch.distributed.barrier()

        if self.rank == 0:
            print_model_size(actor_module)

        log_gpu_memory_usage('After init from HF AutoModel', logger=logger)

        # We wrap FSDP for rollout as well
        mixed_precision_config = fsdp_config.get('mixed_precision', None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get('param_dtype', 'bf16'))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get('reduce_dtype', 'fp32'))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get('buffer_dtype', 'fp32'))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        if self._is_ref:
            mixed_precision = None

        auto_wrap_policy = get_fsdp_wrap_policy(module=actor_module, config=fsdp_config.get('wrap_policy', None))

        if self._is_rollout and self.config.rollout.name == 'hf':
            # TODO(zhangchi.usc1992, shengguangming) fix me. Current, auto_wrap_policy causes HFRollout to hang in Gemma
            auto_wrap_policy = None

        print(f'wrap_policy: {auto_wrap_policy}')

        # TODO(sgm): support hybrid
        if auto_wrap_policy is None:
            sharding_strategy = ShardingStrategy.SHARD_GRAD_OP
        else:
            sharding_strategy = ShardingStrategy.FULL_SHARD

        # TODO: add transformer policy
        actor_module_fsdp = FSDP(
            actor_module,
            param_init_fn=init_fn,
            use_orig_params=False,
            auto_wrap_policy=auto_wrap_policy,
            device_id=torch.cuda.current_device(),
            sharding_strategy=sharding_strategy,  # zero3
            mixed_precision=mixed_precision,
            sync_module_states=True,
            device_mesh=self.device_mesh,
            forward_prefetch=False)

        log_gpu_memory_usage('After Actor FSDP init', logger=logger)

        # TODO: add more optimizer args into config
        if self._is_actor:
            def fresh_optimizer():
                return optim.AdamW(actor_module_fsdp.parameters(),
                                          lr=optim_config.lr,
                                          betas=optim_config.get('betas', (0.9, 0.999)),
                                          weight_decay=optim_config.get('weight_decay', 1e-2))
            if os.environ.get('SEARCH_G1_PROFILE') == 'common8':
                from reproduction.common8.scheduler import collective_actor_initialization
                from search_g1.full_state_restore import optimizer_construction_config
                construction_config=optimizer_construction_config(optim_config,model_path)
                actor_optimizer, actor_lr_scheduler = collective_actor_initialization(fresh_optimizer, construction_config,
                    execution_role=os.environ.get('SEARCH_G1_CALIBRATION_ROLE', 'production'))
            else:
                from verl.utils.weight_continuation import scheduler_from_completed
                actor_optimizer = fresh_optimizer()
                total_steps = optim_config.get('total_training_steps', 0)
                num_warmup_steps_ratio = optim_config.get('lr_warmup_steps_ratio', 0.)
                num_warmup_steps = int(num_warmup_steps_ratio * total_steps)
                actor_lr_scheduler = scheduler_from_completed(
                    actor_optimizer, num_warmup_steps, optim_config.get('completed_updates', 0))
        else:
            actor_optimizer = None
            actor_lr_scheduler = None

        log_gpu_memory_usage('After actor optimizer init', logger=logger)

        return actor_module_fsdp, actor_optimizer, actor_lr_scheduler, actor_model_config

    def _build_rollout(self):
        from torch.distributed.device_mesh import init_device_mesh
        # TODO(sgm): support FSDP hybrid shard for larger model
        infer_tp = self.config.rollout.tensor_model_parallel_size
        dp = self.world_size // infer_tp
        assert self.world_size % infer_tp == 0, f'rollout world_size: {self.world_size} is not divisible by infer_tp: {infer_tp}'
        rollout_device_mesh = init_device_mesh('cuda', mesh_shape=(dp, infer_tp), mesh_dim_names=['dp', 'infer_tp'])

        if self.config.rollout.name == 'hf':
            from verl.workers.rollout import HFRollout
            from verl.workers.sharding_manager import BaseShardingManager
            rollout = HFRollout(module=self.actor_module_fsdp, config=self.config.rollout)
            rollout_sharding_manager = BaseShardingManager()
            # TODO: a sharding manager that do nothing?
        elif self.config.rollout.name == 'vllm':
            from verl.workers.rollout.vllm_rollout import vLLMRollout
            from verl.workers.sharding_manager import FSDPVLLMShardingManager
            log_gpu_memory_usage('Before building vllm rollout', logger=None)
            rollout = vLLMRollout(actor_module=self.actor_module_fsdp,
                                  config=self.config.rollout,
                                  tokenizer=self.tokenizer,
                                  model_hf_config=self.actor_model_config)
            log_gpu_memory_usage('After building vllm rollout', logger=None)
            if torch.distributed.get_world_size() == 1:
                self.config.rollout.load_format = 'dummy_hf'
            rollout_sharding_manager = FSDPVLLMShardingManager(module=self.actor_module_fsdp,
                                                               inference_engine=rollout.inference_engine,
                                                               model_config=self.actor_model_config,
                                                               full_params='hf' in self.config.rollout.load_format,
                                                               device_mesh=rollout_device_mesh,
                                                               reuse_snapshot_weights=self.config.rollout.get('reuse_snapshot_weights', False))
            log_gpu_memory_usage('After building sharding manager', logger=None)

        return rollout, rollout_sharding_manager

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def prefix20_probe_operation(self, operation, payload):
        from search_g1.prefix20_probe_worker import worker_operation
        return worker_operation(self, operation, payload)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def qprobe_operation(self, operation, payload):
        # Driver calls the original allocated rank7 handle directly, so this
        # registered endpoint requires no extra GPU actor or collective call.
        from search_g1.qprobe_sequence_rpc import worker_operation
        return worker_operation(self, operation, payload)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        from verl.workers.actor import DataParallelPPOActor
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get('external_lib', None))

        from omegaconf import OmegaConf
        override_model_config = OmegaConf.to_container(self.config.model.get('override_config', OmegaConf.create()))

        use_remove_padding = self.config.model.get('use_remove_padding', False)

        if self._is_actor or self._is_rollout:
            from verl.utils.weight_continuation import model_initialization_path
            # we need the model for actor and rollout
            if self._is_actor:
                optim_config = self.config.actor.optim
                fsdp_config = self.config.actor.fsdp_config
            else:
                optim_config = None
                fsdp_config = OmegaConf.create()
            self.actor_module_fsdp, self.actor_optimizer, self.actor_lr_scheduler, self.actor_model_config = self._build_model_optimizer(
                model_path=model_initialization_path(self.config.model, actor=True),
                fsdp_config=fsdp_config,
                optim_config=optim_config,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                enable_gradient_checkpointing=self.config.model.get('enable_gradient_checkpointing', False),
                trust_remote_code=self.config.model.get('trust_remote_code', False))

            # get the original unwrapped module
            self.actor_module = self.actor_module_fsdp._fsdp_wrapped_module

            if self._is_offload_param:
                # param is require during state_dict in sharding manager
                offload_fsdp_grad(module=self.actor_module_fsdp)
                log_gpu_memory_usage('After offload actor grad during init', logger=logger)
            if self._is_offload_optimizer:
                offload_fsdp_optimizer(optimizer=self.actor_optimizer)
                log_gpu_memory_usage('After offload actor optimizer during init', logger=logger)
        # load from checkpoint
        if self._is_actor:
            OmegaConf.set_struct(self.config.actor, True)
            with open_dict(self.config.actor):
                self.config.actor.use_remove_padding = use_remove_padding
            self.actor = DataParallelPPOActor(config=self.config.actor,
                                              actor_module=self.actor_module_fsdp,
                                              actor_optimizer=self.actor_optimizer)

        if self._is_rollout:
            self.rollout, self.rollout_sharding_manager = self._build_rollout()

        if self._is_ref:
            from verl.utils.weight_continuation import model_initialization_path
            self.ref_module_fsdp = self._build_model_optimizer(model_path=model_initialization_path(self.config.model, actor=False),
                                                               fsdp_config=self.config.ref.fsdp_config,
                                                               optim_config=None,
                                                               override_model_config=override_model_config,
                                                               use_remove_padding=use_remove_padding,
                                                               trust_remote_code=self.config.model.get(
                                                                   'trust_remote_code', False))[0]
            if self._is_offload_param:
                offload_fsdp_param_and_grad(module=self.ref_module_fsdp, offload_grad=self._is_offload_grad)

            OmegaConf.set_struct(self.config.ref, True)
            with open_dict(self.config.ref):
                self.config.ref.use_remove_padding = use_remove_padding
            self.ref_policy = DataParallelPPOActor(config=self.config.ref, actor_module=self.ref_module_fsdp)

        if self._is_actor:
            self.flops_counter = FlopsCounter(self.actor_model_config)

        from search_g1.prefix20_restore_only import install_mutation_guards
        install_mutation_guards(self)
        torch.cuda.empty_cache()

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data: DataProto):
        from search_g1.prefix20_restore_only import deny_training
        deny_training()
        self._assert_generation_phase_inactive()
        data = data.to(torch.device('cuda', torch.cuda.current_device()))

        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.actor_module_fsdp,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=torch.cuda.current_device())

        data.batch = data.batch.cuda(device=torch.cuda.current_device())

        log_gpu_memory_usage('Before update policy', logger=logger)

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            scheduler_evidence = None
            commit_started = False
            try:
                if os.environ.get('SEARCH_G1_PROFILE') == 'common8':
                    from reproduction.common8.scheduler import OuterSchedulerEvidence, collective_call
                    def begin_scheduler():
                        nonlocal scheduler_evidence
                        if getattr(self.actor_lr_scheduler, '_common8_schedule_kind', None) != 'fresh200_warmup57':
                            return dict(enabled=False,
                                        schedule_kind=getattr(self.actor_lr_scheduler, '_common8_schedule_kind', None))
                        if not hasattr(self, '_outer_scheduler_evidence'):
                            from reproduction.common8.topology import execution_job_id
                            self._outer_scheduler_evidence = OuterSchedulerEvidence(
                                os.environ['SEARCH_G1_BASE'], self.actor_optimizer, self.actor_lr_scheduler,
                                rank=torch.distributed.get_rank(), world_size=torch.distributed.get_world_size(),
                                job_id=execution_job_id())
                        scheduler_evidence = self._outer_scheduler_evidence
                        scheduler_evidence.begin_outer(data.meta_info['logical_update'],
                            optimizer_writes=getattr(self.actor, '_optimizer_writes', 0),
                            batch_gate_sha256=data.meta_info['behavior_probability_gate']['sha256'],
                            **({key:data.meta_info[key] for key in ('attempt_id','attempt_index')}
                               if scheduler_evidence.recoverable else {}))
                        return scheduler_evidence.agreement_state()
                    collective_call(begin_scheduler, stage='begin')
                # Every minibatch guard finishes inside update_policy. A rejected
                # batch can retain real Adam writes, but never advances warmup.
                with Timer(name='update_policy', logger=None) as timer:
                    metrics = self.actor.update_policy(data=data)
                delta_time = timer.last
                attempt_result=getattr(self.actor,'_last_attempt_result',None)
                if attempt_result is not None and attempt_result['status']=='recoverable_guard':
                    if scheduler_evidence is None or not scheduler_evidence.recoverable:
                        raise RuntimeError('Recoverable actor result lacks its explicit formal scheduler')
                    def abort_scheduler_attempt():
                        schedule=scheduler_evidence.abort_attempt(
                            optimizer_writes=getattr(self.actor,'_optimizer_writes',0),
                            opportunities_processed=attempt_result['opportunities_processed'],
                            guard_stage=attempt_result['guard_stage'],guard_reasons=attempt_result['guard_reasons'])
                        return scheduler_evidence.agreement_state()
                    collective_call(abort_scheduler_attempt,stage='abort_attempt')
                    metrics['actor/scheduler_advances_this_outer']=0
                    metrics['actor/completed_outer']=scheduler_evidence.completed_outer
                elif scheduler_evidence is not None:
                    def prepare_scheduler():
                        global_num_tokens = data.meta_info['global_token_num']
                        estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
                        metrics['mfu/actor'] = estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size
                        scheduler_evidence.prepare_outer(optimizer_writes=getattr(self.actor, '_optimizer_writes', 0),
                            **(dict(opportunities_processed=attempt_result['opportunities_processed'])
                               if scheduler_evidence.recoverable else {}))
                        return scheduler_evidence.agreement_state()
                    collective_call(prepare_scheduler, stage='prepare')
                    commit_started = True
                    schedule = None
                    def commit_scheduler():
                        nonlocal schedule
                        schedule = scheduler_evidence.commit_prepared()
                        return scheduler_evidence.agreement_state()
                    collective_call(commit_scheduler, stage='commit')
                    metrics['actor/lr_consumed'] = schedule['consumed_learning_rates'][0]
                    metrics['actor/completed_outer'] = schedule['completed_outer_after']
                    metrics['actor/scheduler_advances_this_outer'] = schedule['scheduler_advances']
                else:
                    # Frozen constant-LR diagnostics and other legacy profiles.
                    global_num_tokens = data.meta_info['global_token_num']
                    estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
                    metrics['mfu/actor'] = estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size
                    self.actor_lr_scheduler.step()
            except BaseException as error:
                if scheduler_evidence is not None:
                    try:
                        if commit_started:
                            scheduler_evidence.fail_after_commit(
                                optimizer_writes=getattr(self.actor, '_optimizer_writes', 0), reason=type(error).__name__)
                        elif scheduler_evidence.pending is not None:
                            scheduler_evidence.reject_outer(
                                optimizer_writes=getattr(self.actor, '_optimizer_writes', 0), reason=type(error).__name__)
                    except BaseException as receipt_error:
                        # Failure to persist must not disguise a committed
                        # scheduler as an unchanged brake rejection.
                        fallback = scheduler_evidence.failure_snapshot(
                            optimizer_writes=getattr(self.actor, '_optimizer_writes', 0),
                            reason=type(error).__name__+'; receipt='+type(receipt_error).__name__,
                            commit_started=commit_started)
                        # Separate console channel retains actual counters if
                        # both durable COMPLETE and fatal ledger append fail.
                        print('SEARCH_G1_FATAL_SCHEDULER_EVIDENCE='+
                              __import__('json').dumps(fallback,allow_nan=False),flush=True)
                        logger.error('Fatal scheduler evidence failure after %s: %s',
                                     'commit' if commit_started else 'rejection', type(receipt_error).__name__)
                raise
            lr = self.actor_lr_scheduler.get_last_lr()[0]
            metrics['actor/lr'] = lr

            if getattr(self.actor_module,'_search_g1_flash3_installed',False):
                from search_g1.flash3_attention import write_execution_audit
                report=write_execution_audit(self.actor_module,rank=self.rank,
                    logical_update=int(data.meta_info['logical_update']),optimizer_writes=self.actor._optimizer_writes)
                metrics['infra/flash3_forward_returns']=sum(report['forward_success_returns'].values())
                metrics['infra/flash3_backward_returns']=report['kernel_backward_success_returns']

            log_gpu_memory_usage('After update policy', logger=logger)

            # TODO: here, we should return all metrics
            if attempt_result is None:
                output = DataProto(meta_info={'metrics': metrics})
            else:
                import numpy as np
                attempt_result=dict(attempt_result,
                    scheduler_advances=metrics['actor/scheduler_advances_this_outer'],
                    completed_outer=scheduler_evidence.completed_outer)
                output = DataProto(meta_info={'metrics': metrics},
                    non_tensor_batch={'recoverable_outer_result':np.array([attempt_result],dtype=object)})

            output = self.ulysses_sharding_manager.postprocess_data(data=output)
            output = output.to('cpu')

        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.actor_module_fsdp, offload_grad=self._is_offload_grad)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)
        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_log_prob(self, data: DataProto) -> DataProto:
        from search_g1.prefix20_restore_only import deny_training
        deny_training()
        self._assert_generation_phase_inactive()
        """mostly copying from generate_sequences"""
        independent_meta = {}
        if os.environ.get('SEARCH_G1_PROFILE') == 'common8':
            from reproduction.common8.behavior_runtime import rollout_logprob_metadata
            independent_meta = rollout_logprob_metadata(self.config.rollout)
            data.meta_info.update(independent_meta)
        data = data.to(torch.device('cuda', torch.cuda.current_device()))

        assert self._is_rollout
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.actor_module_fsdp,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)

        data.batch = data.batch.cuda(device=torch.cuda.current_device())
        meta_info = {'eos_token_id': self.tokenizer.eos_token_id, 'pad_token_id': self.tokenizer.pad_token_id}
        data.meta_info.update(meta_info)

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            old_log_probs = self.actor.compute_log_prob(data=data)
            output = DataProto.from_dict(tensors={'old_log_probs': old_log_probs}, meta_info=independent_meta)
            output = self.ulysses_sharding_manager.postprocess_data(output)

        output = output.to('cpu')

        if self._is_offload_param:
            # NOTE(sgm): the grad is already in CPU, only offload param here
            offload_fsdp_param_and_grad(module=self.actor_module_fsdp, offload_grad=self._is_offload_grad)
        # clear kv cache
        torch.cuda.empty_cache()
        log_gpu_memory_usage('After recompute log prob', logger=logger)
        return output

    def _assert_generation_phase_inactive(self, *, release_snapshot_weights=True):
        if (getattr(self, '_resident_generation_phase', None) is not None or
                getattr(self, '_generation_phase_setup_pending', False)):
            raise RuntimeError('Actor computation cannot overlap a frozen generation phase')
        # No cached sampling copy may survive an actor forward/update/restore.
        # RNG-only chunk boundary captures explicitly retain it.
        manager = getattr(self, 'rollout_sharding_manager', None)
        if release_snapshot_weights and getattr(manager, 'reuse_snapshot_weights', False):
            manager.release_snapshot_weights()

    def _generation_boundary_agreement(self, value):
        gathered = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered, value)
        if any(item != gathered[0] for item in gathered) or value.get('error') is not None:
            raise RuntimeError('Ranks disagree or rejected the generation phase boundary')

    def _generation_setup_agreement(self, error, stage):
        errors = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(errors, error)
        if any(value is not None for value in errors):
            raise RuntimeError('Generation phase setup rejected at '+stage)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def begin_generation_phase(self, snapshot_meta):
        from search_g1.prefix20_restore_only import deny_training
        deny_training()
        import copy
        import hashlib
        import json
        error = None
        if type(snapshot_meta) is dict and all(type(snapshot_meta.get(k)) is str and snapshot_meta[k]
                                               for k in ('phase_id', 'policy_snapshot_id')):
            self._generation_phase_attempt_handle = {key: snapshot_meta[key] for key in ('phase_id', 'policy_snapshot_id')}
            if getattr(self, '_resident_generation_phase', None) is None:
                self._generation_phase_sync_before = self.rollout_sharding_manager.phase_weight_sync_count
                self._generation_phase_actor_offloaded = False
        try:
            self._assert_generation_phase_inactive(release_snapshot_weights=False)
            assert self._is_rollout and os.environ.get('SEARCH_G1_PROFILE') == 'common8'
            assert self.config.rollout.name == 'vllm' and self.config.rollout.tensor_model_parallel_size == 1
            assert type(snapshot_meta) is dict
            for key in ('phase_id', 'policy_snapshot_id'):
                assert type(snapshot_meta[key]) is str and snapshot_meta[key]
            identity = snapshot_meta['behavior_identity']
            reuse = snapshot_meta.get('snapshot_weight_reuse', False)
            retain = snapshot_meta.get('retain_snapshot_weights', False)
            assert type(reuse) is bool and type(retain) is bool and (not retain or reuse)
            assert not reuse or self.config.rollout.get('reuse_snapshot_weights', False) is True
            assert identity['policy_snapshot_id'] == snapshot_meta['policy_snapshot_id']
            provenance = self.rollout.inference_engine.llm_engine._common8_generation_provenance
            assert identity['run_id'] == provenance['run_id'] and identity['slurm_job_id'] == provenance['slurm_job_id']
            source_sha = hashlib.sha256(json.dumps(identity['source_sha256'],sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
            assert source_sha == provenance['source_map_sha256']
            identity_sha = hashlib.sha256(json.dumps(snapshot_meta,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
        except Exception as failure:
            error = type(failure).__name__
            identity_sha = None
        self._generation_boundary_agreement(dict(identity_sha256=identity_sha, error=error))
        needs_weight_sync = True
        if getattr(self.rollout_sharding_manager, 'reuse_snapshot_weights', False):
            cache_error = None
            try:
                needs_weight_sync = self.rollout_sharding_manager.phase_needs_weight_sync(
                    snapshot_meta['policy_snapshot_id'], reuse=reuse)
            except Exception as failure:
                cache_error = type(failure).__name__
            self._generation_boundary_agreement(dict(weight_sync_required=needs_weight_sync, error=cache_error))
        self._generation_phase_setup_pending = True
        preload_error = None
        try:
            if self._is_offload_param and needs_weight_sync:
                self._generation_phase_params_loaded = True
                load_fsdp_param_and_grad(module=self.actor_module_fsdp, device_id=torch.cuda.current_device(), load_grad=self._is_offload_grad)
        except Exception as failure:
            preload_error = type(failure).__name__
        self._generation_setup_agreement(preload_error, 'actor_preload')
        sync_error = None
        try:
            if reuse:
                self.rollout_sharding_manager.begin_phase(snapshot_meta['policy_snapshot_id'],
                    reuse=True, retain_weights=retain)
            else:
                self.rollout_sharding_manager.begin_phase(snapshot_meta['policy_snapshot_id'])
        except Exception as failure:
            sync_error = type(failure).__name__
        self._generation_setup_agreement(sync_error, 'weight_sync')
        begin_error = None
        try:
            if self._is_offload_param and needs_weight_sync:
                # The vLLM loader owns its sampling parameters after sync. Move
                # training shards once, before KV-cache initialization; no
                # actor parameter transitions occur in rank-local requests.
                offload_fsdp_param_and_grad(module=self.actor_module_fsdp, offload_grad=self._is_offload_grad)
                self._generation_phase_params_loaded = False
                self._generation_phase_actor_offloaded = True
            elif self._is_offload_param:
                self._generation_phase_actor_offloaded = True
            self.rollout.begin_generation_phase(snapshot_meta)
        except Exception as failure:
            begin_error = type(failure).__name__
        self._generation_setup_agreement(begin_error, 'engine_setup')
        self._resident_generation_phase = copy.deepcopy(snapshot_meta)
        return {key: snapshot_meta[key] for key in ('phase_id', 'policy_snapshot_id')}

    @register(dispatch_mode=Dispatch.DP_COMPUTE, blocking=False)
    @torch.no_grad()
    def advance_generation_phase(self, handle, new_requests, max_steps=8):
        from search_g1.prefix20_restore_only import deny_training
        deny_training()
        phase = getattr(self, '_resident_generation_phase', None)
        if phase is None or handle != {key: phase[key] for key in ('phase_id', 'policy_snapshot_id')}:
            raise ValueError('Rank-local requests do not belong to the active phase')
        # No FSDP/TP preprocessing, synchronization or probability recompute here.
        return self.rollout.advance_generation_phase(handle, new_requests, max_steps=max_steps)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def end_generation_phase(self, handle, abort=False):
        from search_g1.prefix20_restore_only import deny_training
        deny_training()
        phase = getattr(self, '_resident_generation_phase', None)
        error = None
        try:
            assert type(abort) is bool
            expected = ({key: phase[key] for key in ('phase_id', 'policy_snapshot_id')} if phase is not None
                        else getattr(self, '_generation_phase_attempt_handle', None))
            assert handle == expected and (phase is not None or abort)
            if not abort:
                assert not self.rollout._generation_phase['service'].pending
        except (AssertionError,KeyError,TypeError) as failure:
            error = type(failure).__name__
        self._generation_boundary_agreement(dict(handle=handle, abort=abort, error=error))
        result = (self.rollout.end_generation_phase(handle, abort=abort)
                  if self.rollout._generation_phase is not None or self.rollout._generation_phase_setup_handle is not None
                  else dict(completed_requests=0, engine_steps=0, already_closed=True))
        if self.rollout_sharding_manager._phase_setup_snapshot is not None:
            if getattr(self.rollout_sharding_manager, 'reuse_snapshot_weights', False):
                self.rollout_sharding_manager.end_phase(handle['policy_snapshot_id'], abort=abort)
            else:
                self.rollout_sharding_manager.end_phase(handle['policy_snapshot_id'])
        if self._is_offload_param and getattr(self, '_generation_phase_params_loaded', False):
            offload_fsdp_param_and_grad(module=self.actor_module_fsdp, offload_grad=self._is_offload_grad)
            self._generation_phase_params_loaded = False
        self._resident_generation_phase = None
        self._generation_phase_setup_pending = False
        torch.cuda.empty_cache()
        syncs = self.rollout_sharding_manager.phase_weight_sync_count - self._generation_phase_sync_before
        result.update(phase_id=handle['phase_id'], policy_snapshot_id=handle['policy_snapshot_id'], weight_syncs=syncs,
                      actor_params_offloaded_during_phase=self._generation_phase_actor_offloaded, aborted=abort)
        return result

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto):
        from search_g1.prefix20_restore_only import deny_training
        deny_training()
        if os.environ.get('SEARCH_G1_PROFILE') == 'common8':
            smoke = prompts.meta_info.get('common8_model_smoke') is True
            if smoke and prompts.meta_info.get('do_sample') is not False:
                raise ValueError('common8 model-smoke exception requires explicit greedy mode')
            if not smoke and prompts.meta_info.get('recompute_log_prob') is not False:
                raise ValueError('common8 requires explicit recompute_log_prob=False before rollout; driver audits actual behavior first')
        self._assert_generation_phase_inactive()
        prompts = prompts.to(torch.device('cuda', torch.cuda.current_device()))
        # set to False if it is validation
        recompute_log_prob = prompts.meta_info.get('recompute_log_prob', True)

        assert self._is_rollout
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.actor_module_fsdp,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)

        prompts.batch = prompts.batch.cuda(device=torch.cuda.current_device())
        meta_info = {'eos_token_id': self.tokenizer.eos_token_id, 'pad_token_id': self.tokenizer.pad_token_id}
        prompts.meta_info.update(meta_info)
        with self.rollout_sharding_manager:
            log_gpu_memory_usage('After entering rollout sharding manager', logger=logger)

            prompts = self.rollout_sharding_manager.preprocess_data(prompts)
            output = self.rollout.generate_sequences(prompts=prompts)

            log_gpu_memory_usage('After rollout generation', logger=logger)

            output = self.rollout_sharding_manager.postprocess_data(output)

        if self._is_actor and recompute_log_prob:
            # we should always recompute old_log_probs when it is HybridEngine
            output.meta_info['micro_batch_size'] = self.config.rollout.log_prob_micro_batch_size
            output.meta_info['max_token_len'] = self.config.rollout.log_prob_max_token_len_per_gpu
            output.meta_info['use_dynamic_bsz'] = self.config.rollout.log_prob_use_dynamic_bsz
            output.meta_info['temperature'] = self.config.rollout.temperature
            # perform recompute log_prob
            with self.ulysses_sharding_manager:
                output = self.ulysses_sharding_manager.preprocess_data(output)
                old_log_probs = self.actor.compute_log_prob(data=output)
                output.batch['old_log_probs'] = old_log_probs
                output = self.ulysses_sharding_manager.postprocess_data(output)

        output = output.to('cpu')

        if self._is_offload_param:
            # NOTE(sgm): the grad is already in CPU, only offload param here
            offload_fsdp_param_and_grad(module=self.actor_module_fsdp, offload_grad=self._is_offload_grad)
        # clear kv cache
        torch.cuda.empty_cache()
        log_gpu_memory_usage('After recompute log prob', logger=logger)
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_ref_log_prob(self, data: DataProto):
        self._assert_generation_phase_inactive()
        assert self._is_ref

        data = data.to(torch.device('cuda', torch.cuda.current_device()))

        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.ref_module_fsdp,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)

        micro_batch_size = self.config.ref.log_prob_micro_batch_size
        data.meta_info['micro_batch_size'] = micro_batch_size
        data.meta_info['temperature'] = self.config.rollout.temperature
        data.meta_info['max_token_len'] = self.config.ref.log_prob_max_token_len_per_gpu
        data.meta_info['use_dynamic_bsz'] = self.config.ref.log_prob_use_dynamic_bsz
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            output = self.ref_policy.compute_log_prob(data=data)
            output = DataProto.from_dict(tensors={'ref_log_prob': output})
            output = self.ulysses_sharding_manager.postprocess_data(output)

        output = output.to('cpu')

        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.ref_module_fsdp, offload_grad=self._is_offload_grad)
        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def capture_mixed_state(self, directory, checkpoint=False):
        from search_g1.prefix20_restore_only import deny_training
        deny_training()
        self._assert_generation_phase_inactive(release_snapshot_weights=checkpoint)
        assert self._is_actor and self._is_rollout
        from search_g1.training_state import capture_worker
        return capture_worker(self,directory,checkpoint=checkpoint)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def capture_frozen_teacher_boundary(self, directory):
        self._assert_generation_phase_inactive()
        assert self._is_actor and self._is_rollout
        from search_g1.qprobe_teacher_worker import capture
        return capture(self, directory)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def restore_frozen_teacher_rng(self, directory):
        assert self._is_actor and self._is_rollout
        from search_g1.qprobe_teacher_worker import restore_rng
        return restore_rng(self, directory)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def restore_mixed_full_state(self, receipt_path):
        self._assert_generation_phase_inactive()
        assert self._is_actor and self._is_rollout
        from search_g1.full_state_restore import restore_worker
        from search_g1.prefix20_restore_only import active, assert_target_receipt
        if active():assert_target_receipt(receipt_path)
        return restore_worker(self, receipt_path)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def readback_mixed_full_state(self, receipt_path, phase):
        from search_g1.prefix20_restore_only import readback_actor
        return readback_actor(self, receipt_path, phase)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None):
        from search_g1.prefix20_restore_only import deny_training
        deny_training()
        self._assert_generation_phase_inactive()
        assert self._is_actor
        import torch
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.actor_module_fsdp,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)

        # TODO: support DCP and save sharded checkpoints
        import torch.distributed
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.actor.actor_module, StateDictType.FULL_STATE_DICT, cfg):
            state_dict = self.actor.actor_module.state_dict()
        if self.rank == 0:
            print(f'Saving actor checkpoint to {local_path}')
            os.makedirs(local_path, exist_ok=True)
            self.actor_module.save_pretrained(local_path, state_dict=state_dict)
            self.tokenizer.save_pretrained(local_path)
            if hdfs_path is not None:
                print(f'Uploading actor checkpoint to {hdfs_path}')
                hdfs_io.makedirs(hdfs_path, exist_ok=True)
                hdfs_io.copy(src=local_path, dst=hdfs_path)

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.actor_module_fsdp, offload_grad=self._is_offload_grad)


class CriticWorker(Worker):

    def __init__(self, config):
        super().__init__()
        import torch.distributed
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        self.config = config

        # build device mesh for Ulysses Sequence Parallel
        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh
        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.get('ulysses_sequence_parallel_size', 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh('cuda',
                                                        mesh_shape=(dp, self.ulysses_sequence_parallel_size),
                                                        mesh_dim_names=['dp', 'sp'])

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        # set FSDP offload params
        self._is_offload_param = self.config.model.fsdp_config.param_offload
        self._is_offload_grad = self.config.model.fsdp_config.grad_offload
        self._is_offload_optimizer = self.config.model.fsdp_config.optimizer_offload

        # normalize config
        self.config.ppo_mini_batch_size //= (torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size)
        self.config.ppo_micro_batch_size //= (torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size)
        self.config.forward_micro_batch_size //= (torch.distributed.get_world_size() //
                                                  self.ulysses_sequence_parallel_size)

    def _build_critic_model_optimizer(self, config):
        # the following line is necessary
        from verl.utils.model import LambdaLayer, print_model_size, squeeze
        from verl.utils.torch_dtypes import PrecisionType
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, MixedPrecision
        from torch import optim

        local_path = copy_local_path_from_hdfs(config.model.path)
        # note that the tokenizer between actor and critic may be different. So override tokenizer info with actor info
        # using random initialized model from any architecture. May not be the same as Actor.

        tokenizer_path = copy_local_path_from_hdfs(config.model.tokenizer_path)
        self.tokenizer = hf_tokenizer(tokenizer_path, trust_remote_code=config.model.get('trust_remote_code', False))

        from omegaconf import OmegaConf
        override_config = OmegaConf.to_container(self.config.model.get('override_config', OmegaConf.create()))
        override_config_kwargs = {
            'bos_token_id': self.tokenizer.bos_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'pad_token_id': self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_config)
        if self.rank == 0:
            print(f'Critic overriding config {override_config_kwargs}')

        torch_dtype = self.config.model.fsdp_config.get('model_dtype', 'fp32')
        torch_dtype = PrecisionType.to_dtype(torch_dtype)

        from transformers import AutoConfig, AutoModelForTokenClassification
        from torch import nn

        trust_remote_code = False
        critic_model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)
        critic_model_config.num_labels = 1

        use_remove_padding = config.model.get('use_remove_padding', False)
        if use_remove_padding:
            from verl.models.registry import check_model_support_rmpad
            check_model_support_rmpad(critic_model_config.model_type)

        if use_remove_padding and self.ulysses_sequence_parallel_size > 1:
            from verl.models.transformers.monkey_patch import apply_monkey_patch
            apply_monkey_patch(critic_model_config, verbose=True)

        init_context = get_init_weight_context_manager()
        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            setattr(critic_model_config, 'classifier_dropout', 0.)
            setattr(critic_model_config, 'hidden_dropout', '0')
            critic_module = AutoModelForTokenClassification.from_pretrained(pretrained_model_name_or_path=local_path,
                                                                            torch_dtype=torch_dtype,
                                                                            config=critic_model_config,
                                                                            attn_implementation='sdpa',
                                                                            trust_remote_code=trust_remote_code)

            # some parameters may not in torch_dtype
            critic_module.to(torch_dtype)

            if config.model.get('enable_gradient_checkpointing', False):
                critic_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        if self.rank == 0:
            print_model_size(critic_module)

        self.critic_model_config = critic_model_config

        fsdp_config = self.config.model.fsdp_config
        mixed_precision_config = fsdp_config.get('mixed_precision', None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get('param_dtype', 'bf16'))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get('reduce_dtype', 'fp32'))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get('buffer_dtype', 'fp32'))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        auto_wrap_policy = get_fsdp_wrap_policy(module=critic_module, config=self.config.model.fsdp_config.wrap_policy)

        log_gpu_memory_usage('Before critic FSDP', logger=None)

        critic_module = FSDP(critic_module,
                             param_init_fn=init_fn,
                             use_orig_params=False,
                             auto_wrap_policy=auto_wrap_policy,
                             device_id=torch.cuda.current_device(),
                             sharding_strategy=ShardingStrategy.FULL_SHARD,
                             mixed_precision=mixed_precision,
                             sync_module_states=True,
                             forward_prefetch=False)

        log_gpu_memory_usage('After critic FSDP', logger=None)

        critic_optimizer = optim.AdamW(critic_module.parameters(),
                                       lr=config.optim.lr,
                                       betas=config.optim.get('betas', (0.9, 0.999)),
                                       weight_decay=config.optim.get('weight_decay', 1e-2))

        total_steps = config.optim.get('total_training_steps', 0)
        num_warmup_steps_ratio = config.optim.get('lr_warmup_steps_ratio', 0.)
        num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

        print(f'Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}')

        from verl.utils.torch_functional import get_constant_schedule_with_warmup
        critic_lr_scheduler = get_constant_schedule_with_warmup(optimizer=critic_optimizer,
                                                                num_warmup_steps=num_warmup_steps)

        return critic_module, critic_optimizer, critic_lr_scheduler

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get('external_lib', None))

        from verl.workers.critic import DataParallelPPOCritic
        self.critic_module, self.critic_optimizer, self.critic_lr_scheduler = self._build_critic_model_optimizer(
            self.config)

        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.critic_module, offload_grad=self._is_offload_grad)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.critic_optimizer)

        self.critic = DataParallelPPOCritic(config=self.config,
                                            critic_module=self.critic_module,
                                            critic_optimizer=self.critic_optimizer)

        self.flops_counter = FlopsCounter(self.critic_model_config)

        torch.cuda.empty_cache()

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_values(self, data: DataProto):
        data = data.to('cuda')

        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.critic_module,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)
        micro_batch_size = self.config.forward_micro_batch_size
        data.meta_info['micro_batch_size'] = micro_batch_size
        data.meta_info['max_token_len'] = self.config.forward_max_token_len_per_gpu
        data.meta_info['use_dynamic_bsz'] = self.config.use_dynamic_bsz
        # perform forward computation
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            values = self.critic.compute_values(data=data)
            output = DataProto.from_dict(tensors={'values': values})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        output = output.to('cpu')
        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.critic_module, offload_grad=self._is_offload_grad)
        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_critic(self, data: DataProto):
        data = data.to('cuda')
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.critic_module,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.critic_optimizer, device_id=torch.cuda.current_device())

        # perform forward computation
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)

            with Timer(name='update_critic', logger=None) as timer:
                metrics = self.critic.update_critic(data=data)
            delta_time = timer.last

            global_num_tokens = data.meta_info['global_token_num']
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics['mfu/critic'] = estimated_flops * self.config.ppo_epochs / promised_flops / self.world_size

            self.critic_lr_scheduler.step()
            lr = self.critic_lr_scheduler.get_last_lr()[0]
            metrics['critic/lr'] = lr

            output = DataProto(batch=None, meta_info={'metrics': metrics})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.critic_module, offload_grad=self._is_offload_grad)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.critic_optimizer)
        torch.cuda.empty_cache()
        output = output.to('cpu')
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None):
        import torch
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.critic_module,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)

        # TODO: support DCP and save sharded checkpoints
        import torch.distributed
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.critic_module, StateDictType.FULL_STATE_DICT, cfg):
            state_dict = self.critic_module.state_dict()
        if self.rank == 0:
            print(f'Saving critic checkpoint to {local_path}')
            os.makedirs(local_path, exist_ok=True)
            self.critic_module._fsdp_wrapped_module.save_pretrained(local_path, state_dict=state_dict)
            self.tokenizer.save_pretrained(local_path)
            if hdfs_path is not None:
                print(f'Uploading critic checkpoint to {hdfs_path}')
                hdfs_io.makedirs(hdfs_path, exist_ok=True)
                hdfs_io.copy(src=local_path, dst=hdfs_path)

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.critic_module, offload_grad=self._is_offload_grad)


# TODO(sgm): we may need to extract it to dp_reward_model.py
class RewardModelWorker(Worker):
    """
    Note that we only implement the reward model that is subclass of AutoModelForTokenClassification.
    """

    def __init__(self, config):
        super().__init__()
        import torch.distributed
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        self.config = config

        # build device mesh for Ulysses Sequence Parallel
        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh
        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.get('ulysses_sequence_parallel_size', 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh('cuda',
                                                        mesh_shape=(dp, self.ulysses_sequence_parallel_size),
                                                        mesh_dim_names=['dp', 'sp'])

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        self.use_remove_padding = self.config.model.get('use_remove_padding', False)
        self.config.micro_batch_size //= torch.distributed.get_world_size()

    def _build_model(self, config):
        # the following line is necessary
        from transformers import AutoModelForTokenClassification, AutoConfig
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, CPUOffload

        # download the checkpoint from hdfs
        local_path = copy_local_path_from_hdfs(config.model.path)

        if self.config.model.input_tokenizer is None:
            self._do_switch_chat_template = False
        else:
            self._do_switch_chat_template = True
            input_tokenizer_local_path = copy_local_path_from_hdfs(config.model.input_tokenizer)
            self.input_tokenizer = hf_tokenizer(input_tokenizer_local_path,
                                                trust_remote_code=config.model.get('trust_remote_code', False))
            self.tokenizer = hf_tokenizer(local_path, trust_remote_code=config.model.get('trust_remote_code', False))

        trust_remote_code = config.model.get('trust_remote_code', False)
        model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)
        model_config.num_labels = 1

        use_remove_padding = config.model.get('use_remove_padding', False)
        if use_remove_padding:
            from verl.models.registry import check_model_support_rmpad
            check_model_support_rmpad(model_config.model_type)

        if use_remove_padding and self.ulysses_sequence_parallel_size > 1:
            from verl.models.transformers.monkey_patch import apply_monkey_patch
            apply_monkey_patch(model_config, verbose=True)

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        init_context = get_init_weight_context_manager(use_meta_tensor=not model_config.tie_word_embeddings)

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            setattr(model_config, 'classifier_dropout', 0.)
            reward_module = AutoModelForTokenClassification.from_pretrained(pretrained_model_name_or_path=local_path,
                                                                            config=model_config,
                                                                            torch_dtype=torch.bfloat16,
                                                                            attn_implementation='sdpa',
                                                                            trust_remote_code=trust_remote_code)
            reward_module.to(torch.bfloat16)
        auto_wrap_policy = get_fsdp_wrap_policy(module=reward_module, config=self.config.model.fsdp_config)

        reward_module = FSDP(
            reward_module,
            param_init_fn=init_fn,
            use_orig_params=False,
            auto_wrap_policy=auto_wrap_policy,
            device_id=torch.cuda.current_device(),
            sharding_strategy=ShardingStrategy.FULL_SHARD,  # zero3
            sync_module_states=True,
            cpu_offload=CPUOffload(offload_params=self.config.model.fsdp_config.param_offload),
            forward_prefetch=False)

        return reward_module

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get('external_lib', None))
        self.reward_module = self._build_model(config=self.config)
        torch.cuda.empty_cache()

    def _forward_micro_batch(self, micro_batch):
        from flash_attn.bert_padding import pad_input, unpad_input, index_first_axis, rearrange
        from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad

        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch['attention_mask']
            position_ids = micro_batch['position_ids']

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                           attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                      indices).transpose(0, 1)

                # pad and slice the inputs if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, \
                                                                                                position_ids_rmpad, \
                                                                                                sp_size=self.ulysses_sequence_parallel_size)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.reward_module(input_ids=input_ids_rmpad,
                                            attention_mask=None,
                                            position_ids=position_ids_rmpad,
                                            use_cache=False)  # prevent model thinks we are generating
                reward_rmpad = output.logits
                reward_rmpad = reward_rmpad.squeeze(0)  # (total_nnz)

                # gather output if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    reward_rmpad = gather_outpus_and_unpad(reward_rmpad,
                                                           gather_dim=0,
                                                           unpad_dim=0,
                                                           padding_size=pad_size)

                # pad it back
                rm_score = pad_input(reward_rmpad, indices=indices, batch=batch_size, seqlen=seqlen).squeeze(-1)
            else:
                output = self.reward_module(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            position_ids=position_ids)
                rm_score = output.logits  # (batch_size, seq_len, 1)
                rm_score = rm_score.squeeze(-1)

            # extract the result of the last valid token
            eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (bsz,)
            rm_score = rm_score[torch.arange(batch_size), eos_mask_idx]
            return rm_score

    def _expand_to_token_level(self, data: DataProto, scores: torch.Tensor):
        batch_size = data.batch.batch_size[0]
        # expand as token_level_reward
        attention_mask = data.batch['attention_mask']
        position_ids = data.batch['position_ids']
        response_length = data.batch['responses'].shape[-1]
        eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (bsz,)
        token_level_scores = torch.zeros_like(attention_mask, dtype=scores.dtype)  # (bsz, seqlen)
        token_level_scores[torch.arange(batch_size), eos_mask_idx] = scores

        # select the response part
        token_level_scores = token_level_scores[:, -response_length:]

        return token_level_scores

    def _switch_chat_template(self, data: DataProto):
        src_max_length = data.batch['attention_mask'].shape[-1]

        src_tokenizer = self.input_tokenizer
        target_tokenizer = self.tokenizer

        rm_input_ids = []
        rm_attention_mask = []

        for i in range(data.batch.batch_size[0]):
            # extract raw prompt
            chat: list = data.non_tensor_batch['raw_prompt'][i].tolist()

            # extract response
            response_ids = data.batch['responses'][i]
            response_length = response_ids.shape[-1]
            valid_response_length = data.batch['attention_mask'][i][-response_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            response = src_tokenizer.decode(valid_response_ids)
            # remove bos and eos
            response = response.replace(src_tokenizer.eos_token, '')

            chat.append({'role': 'assistant', 'content': response})

            prompt_with_chat_template = target_tokenizer.apply_chat_template(chat,
                                                                             add_generation_prompt=False,
                                                                             tokenize=False)
            if self.rank == 0 and i == 0:
                # for debugging purpose
                print(f'Switch template. chat: {prompt_with_chat_template}')

            # the maximum length is actually determined by the reward model itself
            max_length = self.config.get('max_length', src_max_length)
            if max_length is None:
                max_length = src_max_length
            input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
                prompt=prompt_with_chat_template,
                tokenizer=target_tokenizer,
                max_length=max_length,
                pad_token_id=target_tokenizer.pad_token_id,
                left_pad=False,  # right padding
                truncation=self.config.get('truncation', 'right'))  # truncate from the right

            rm_input_ids.append(input_ids)
            rm_attention_mask.append(attention_mask)

        rm_input_ids = torch.cat(rm_input_ids, dim=0)
        rm_attention_mask = torch.cat(rm_attention_mask, dim=0)

        rm_position_ids = compute_position_id_with_mask(rm_attention_mask)

        rm_inputs = {'input_ids': rm_input_ids, 'attention_mask': rm_attention_mask, 'position_ids': rm_position_ids}

        return DataProto.from_dict(rm_inputs)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_rm_score(self, data: DataProto):
        import itertools
        from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx
        data = data.to('cuda')
        if self._do_switch_chat_template:
            rm_data = self._switch_chat_template(data)

        rm_data.batch = rm_data.batch.cuda()

        # perform forward computation
        with self.ulysses_sharding_manager:
            rm_data = self.ulysses_sharding_manager.preprocess_data(data=rm_data)
            data = self.ulysses_sharding_manager.preprocess_data(data=data)

            use_dynamic_bsz = self.config.use_dynamic_bsz
            if use_dynamic_bsz:
                max_token_len = self.config.forward_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, indices = rearrange_micro_batches(batch=rm_data.batch, max_token_len=max_token_len)
            else:
                micro_batches = rm_data.batch.split(self.config.micro_batch_size)
            output = []
            for micro_batch in micro_batches:
                rm_score = self._forward_micro_batch(micro_batch)
                output.append(rm_score)
            scores = torch.cat(output, dim=0)  # (batch_size)

            if use_dynamic_bsz:
                indices = list(itertools.chain.from_iterable(indices))
                assert len(indices) == scores.size(0), f"{len(indices)} vs. {scores.size()}"
                revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
                scores = scores[revert_indices]

            token_level_scores = self._expand_to_token_level(data, scores)
            # Note that this is only the scores, may not be the final rewards used to train RL
            output = DataProto.from_dict(tensors={'rm_scores': token_level_scores})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        output = output.to('cpu')
        torch.cuda.empty_cache()
        return output
