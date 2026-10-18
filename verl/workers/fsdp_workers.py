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


def inspect_fresh_worker_state(worker, expected_model_path):
    """Observe empty Adam/scheduler before fresh fit; not a model-byte audit."""
    from pathlib import Path
    from search_r1.baselines.sampling_consistency import collective_device, sync_prepare, check_signature
    from search_r1.baselines.tree_fresh200 import refuse_recovery_environment
    device = collective_device()
    error = None
    try:
        if not worker._is_actor or worker.config.actor.get('fresh200_protocol') != 'tree_fresh200_five_leaf_v1':
            raise ValueError('Fresh initialization inspection requires the explicit fresh200 actor profile')
        refuse_recovery_environment()
        configured = Path(worker.config.model.path)
        expected = Path(expected_model_path)
        if (not configured.is_absolute() or not expected.is_absolute()
                or configured.resolve() != expected.resolve()):
            raise ValueError('Fresh configured model path differs from the driver-bound official path')
        if len(worker.actor_optimizer.state) != 0:
            raise ValueError('Fresh Adam requires the entire state mapping to be empty, not merely step zero')
        scheduler = worker.actor_lr_scheduler.last_epoch
        if type(scheduler) is not int or scheduler != 0:
            raise ValueError('Fresh scheduler must be at actual initial step zero')
        record = dict(rank=torch.distributed.get_rank(), world_size=torch.distributed.get_world_size(),
                      training_protocol='tree_fresh200_five_leaf_v1', calibration_profile='fresh_base_zero_v1',
                      configured_model_path=str(configured.resolve()), expected_model_path=str(expected.resolve()),
                      optimizer_state_entries=len(worker.actor_optimizer.state), optimizer_step_values=[],
                      scheduler_step=scheduler, optimizer_committed=0, scheduler_advanced=False,
                      checkpoint_recovery_environment_absent=True,
                      complete_model_parameter_bytes_verified=False,
                      scope='Actual empty worker Adam and scheduler; configured load path bound to driver official source, no complete model value comparison',
                      code_commit=os.environ['SEARCH_R1_BASELINE_COMMIT'],
                      config_sha256=os.environ['SEARCH_R1_BASELINE_CONFIG_SHA256'],
                      data_identity=os.environ['SEARCH_R1_BASELINE_DATA_IDENTITY'],
                      model_identity=os.environ['SEARCH_R1_MODEL_IDENTITY'],
                      job_id=__import__('search_r1.baselines.execution_identity',fromlist=['execution_id']).execution_id(), run_id=os.environ['SEARCH_R1_BASELINE_RUN_ID'])
    except Exception as exc:
        error = repr(exc)
    sync_prepare(error, device)
    check_signature([record['optimizer_state_entries'], record['scheduler_step']], device)
    return record


def compact_normal_guard_after_actor(worker, refusal):
    """GPU finite checks and tiny receipts only; no model/optimizer CPU snapshot."""
    from search_r1.baselines.tree_guard_continue import normal_guard_summary
    from search_r1.baselines.policy_safety_runtime import PolicySafetyRefusal
    count = worker.actor._actual_committed_optimizer_calls
    error, summary = None, None
    try:
        summary = normal_guard_summary(refusal, count, worker.actor._fresh_opportunity_ledger)
    except Exception as exc:
        error = repr(exc)
    admission = [dict(error=error, writes=count)]
    if torch.distributed.is_initialized():
        admission = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(admission, dict(error=error, writes=count))
    if (any(value['error'] is not None for value in admission) or
            len({value['writes'] for value in admission}) != 1):
        raise PolicySafetyRefusal('Normal guard ledger/count fault: ' + str(admission),
                                  matched_all_ranks=True, safety_guard=False)
    # This scans actual model/moments in place, without copying their state.
    worker.actor._policy_safety.parameters_and_adam('normal_guard_model_adam_before_continue')
    worker.actor_optimizer.zero_grad()
    error = None
    try:
        parameters = list(worker.actor.actor_module.parameters())
        if any(parameter.grad is not None for parameter in parameters):
            raise ValueError('Normal zero_grad did not clear refused gradients')
        private = [getattr(parameter, name) for parameter in parameters
            for name in ('_saved_grad_shard', '_cpu_grad')
            if isinstance(getattr(parameter, name, None), torch.Tensor)]
        if not all(bool(torch.isfinite(value).all()) for value in private):
            raise ValueError('Nonfinite private FSDP gradient remains after normal zero_grad')
    except Exception as exc:
        error = repr(exc)
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    record = dict(summary, rank=rank, normal_zero_grad=(error is None), error=error,
                  model_adam_finite=True, threshold_unchanged=True)
    records = [record]
    if torch.distributed.is_initialized():
        records = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(records, record)
    if (any(value['error'] is not None for value in records) or
            len({value['optimizer_writes'] for value in records}) != 1):
        raise PolicySafetyRefusal('Normal guard continuation state/count fault: ' + str(records),
                                  matched_all_ranks=True, safety_guard=False)
    return dict(status='NORMAL_SAFETY_GUARD_ROUND_CONSUMED', ranks=records,
                model_optimizer_snapshot_saved=False, automatic_rollback=False)


def fresh200_scheduler_after_actor(worker, data, metrics, scheduler_before_outer, lr_before_outer):
    """Admit 20 processed opportunities; opt-in rounds may have zero actual writes.

    Advance once after admission and retain direct evidence. The legacy strict
    branch continues to require all 20 writes and zero skips.

    An incomplete outer retains its real optimizer writes. No rollback or mean
    metric can turn partial progress into completed outer credit.
    """
    import math
    from pathlib import Path
    from search_r1.baselines.sampling_consistency import collective_device
    from search_r1.baselines.training_state import atomic_save
    device = collective_device()
    dist = torch.distributed
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    def global_invalid(error):
        flag = torch.tensor(int(error is not None), device=device)
        if dist.is_initialized(): dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item())
    outer = data.meta_info.get('baseline_outer_step')
    writes = getattr(worker.actor, '_actual_committed_optimizer_calls', None)
    from search_r1.baselines.tree_guard_continue import enabled as guard_continue_enabled, validate_ledger, round_progress_enabled
    continuing_guard = guard_continue_enabled(worker.config.actor)
    ledger = getattr(worker.actor, '_fresh_opportunity_ledger', [])
    returned_writes, skips = metrics.get('actor/optimizer_steps'), metrics.get('actor/skipped_empty_minibatches')
    error = None
    before_lrs = None
    try:
        before_lrs = [float(value) for value in worker.actor_lr_scheduler.get_last_lr()]
        if worker.config.actor.get('fresh200_protocol') != 'tree_fresh200_five_leaf_v1':
            raise ValueError('Fresh scheduler admission requires the explicit fresh200 profile')
        if type(outer) is not int or outer < 1:
            raise ValueError('Fresh outer must be a positive direct driver counter')
        if continuing_guard:
            validate_ledger(ledger, writes, allow_zero=round_progress_enabled(worker.config.actor))
            if returned_writes != writes or metrics.get('actor/optimizer_opportunities_processed') != 20:
                raise ValueError('Fresh completed opportunity accounting differs from actual actor ledger')
        elif (type(writes) is not int or writes != 20 or type(returned_writes) is not int
              or returned_writes != writes or type(skips) is not int or skips != 0):
            raise ValueError('Fresh outer requires each rank to commit 20 optimizer calls and skip zero')
        if (type(scheduler_before_outer) is not int or scheduler_before_outer != outer - 1
                or worker.actor_lr_scheduler.last_epoch != scheduler_before_outer):
            raise ValueError('Fresh actual scheduler-before must equal observed outer minus one')
        if (not before_lrs or not all(math.isfinite(value) and value >= 0 for value in before_lrs)
                or not math.isfinite(float(lr_before_outer)) or before_lrs[0] != lr_before_outer):
            raise ValueError('Fresh actual learning rate changed during actor update or is invalid')
    except Exception as exc:
        error = repr(exc)
    # A valid counter on each rank must also identify the same outer globally.
    signature_values = [outer if type(outer) is int and 0 < outer < 2**53 else -1]
    if continuing_guard:
        signature_values += [writes if type(writes) is int else -1, len(ledger), skips if type(skips) is int else -1]
    signature = torch.tensor(signature_values, dtype=torch.float64, device=device)
    lo, hi = signature.clone(), signature.clone()
    if dist.is_initialized():
        dist.all_reduce(lo, op=dist.ReduceOp.MIN); dist.all_reduce(hi, op=dist.ReduceOp.MAX)
    if not torch.equal(lo, hi): error = error or 'Fresh observed outer differs across ranks'
    global_before_invalid = global_invalid(error)
    step_error = None
    scheduler_step_invoked = False
    if not global_before_invalid:
        scheduler_step_invoked = True
        try:
            worker.actor_lr_scheduler.step()
        except Exception as exc:
            step_error = repr(exc)
    after = worker.actor_lr_scheduler.last_epoch
    after_lrs = None
    try:
        after_lrs = [float(value) for value in worker.actor_lr_scheduler.get_last_lr()]
        if not after_lrs or not all(math.isfinite(value) and value >= 0 for value in after_lrs):
            raise ValueError('Fresh scheduler-after learning rate invalid')
        if not global_before_invalid and (type(after) is not int or after != outer):
            raise ValueError('Fresh scheduler-after must equal the completed observed outer')
        if global_before_invalid and after != scheduler_before_outer:
            raise ValueError('Refused fresh scheduler advanced before admission')
    except Exception as exc:
        step_error = step_error or repr(exc)
    global_after_invalid = global_invalid(step_error)
    record = dict(rank=rank, world_size=world, training_protocol='tree_fresh200_five_leaf_v1',
                  observed_outer=outer if type(outer) is int else None,
                  actual_optimizer_calls=writes if type(writes) is int else None,
                  opportunities_processed=len(ledger) if continuing_guard else 20,
                  opportunities_skipped=sum(e['actual_writes']==0 for e in ledger) if continuing_guard else 0,
                  optimizer_opportunity_ledger=ledger if continuing_guard else None,
                  skipped_empty_minibatches=skips if type(skips) is int else None,
                  returned_optimizer_calls=returned_writes if type(returned_writes) is int else None,
                  direct_count_source='actor optimizer.step successful-return counter before any publish/audit IO',
                  scheduler_step_before=scheduler_before_outer, scheduler_step_after=after,
                  scheduler_step_invoked=scheduler_step_invoked,
                  learning_rates_before=before_lrs, learning_rates_after=after_lrs,
                  first_outer_lr_zero=(outer == 1 and lr_before_outer == 0),
                  zero_lr_optimizer_calls_are_real=True, parameter_delta_verified_by_this_record=False,
                  all_rank_before_admitted=not global_before_invalid,
                  all_rank_scheduler_after_valid=not global_after_invalid,
                  local_before_error=error, local_scheduler_error=step_error,
                  status=('REFUSED_BEFORE_SCHEDULER_WRITES_RETAINED' if global_before_invalid
                          else 'SCHEDULER_AFTER_REFUSAL_WRITES_RETAINED' if global_after_invalid
                          else 'DIRECT_RANK_COMPLETE_OUTER_OBSERVED'),
                  code_commit=os.environ.get('SEARCH_R1_BASELINE_COMMIT'),
                  config_sha256=os.environ.get('SEARCH_R1_BASELINE_CONFIG_SHA256'),
                  data_identity=os.environ.get('SEARCH_R1_BASELINE_DATA_IDENTITY'),
                  model_identity=os.environ.get('SEARCH_R1_MODEL_IDENTITY'),
                  job_id=os.environ.get('SLURM_JOB_ID'), run_id=os.environ.get('SEARCH_R1_BASELINE_RUN_ID'),
                  scope='Direct rank optimizer/scheduler accounting, no probability or quality acceptance')
    io_error = None
    try:
        if any(record[key] is None for key in ('code_commit','config_sha256','data_identity','model_identity','job_id','run_id')):
            raise ValueError('Missing fresh rank source/run identity')
        from search_r1.baselines.tree_guard_continue import evidence_suffix
        path = Path(os.environ['SEARCH_R1_BASELINE_OUTPUT'])/'fresh-state'/f'outer-{outer:06d}{evidence_suffix(data.meta_info)}-rank{rank:02d}.json'
        record['attempt_id'] = data.meta_info.get('baseline_attempt_id')
        record_sha = atomic_save(record, path, json_format=True)
    except Exception as exc:
        io_error = repr(exc)
    global_io_invalid = global_invalid(io_error)
    if global_before_invalid or global_after_invalid or global_io_invalid:
        from search_r1.baselines.policy_safety_runtime import PolicySafetyRefusal
        raise PolicySafetyRefusal('Global fresh outer scheduler/accounting refusal; actual optimizer writes retained; local: '
                                  +str(error or step_error or io_error), matched_all_ranks=True)
    metrics['actor/fresh_direct_rank_outer_admitted'] = 1
    return dict(path=str(path), sha256=record_sha, rank=rank, actual_optimizer_calls=writes,
                scheduler_step_before=scheduler_before_outer, scheduler_step_after=after)


class ActorRolloutRefWorker(Worker):
    """
    This worker can be instantiated as a standalone actor or a standalone rollout or a standalone reference policy
    or a hybrid engine based on the config.rollout
    """

    def __init__(self, config: DictConfig, role: str):
        super().__init__()
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
        if (self.config.actor.get('baseline_method') == 'tree_grpo' and
                self.config.actor.get('policy_safety', {}).get('sampling_probability_correction', {}).get('enabled') is True):
            from search_r1.baselines.tree_raw_tokens import register_validation_tokenizer
            self._tree_raw_validation_tokenizer_identity = register_validation_tokenizer(self.tokenizer)

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
                                                                attn_implementation='flash_attention_2',
                                                                trust_remote_code=trust_remote_code)
            # some parameters may not in torch_dtype. TODO(zhangchi.usc1992) remove this after we switch to fsdp2
            actor_module.to(torch_dtype)

            if enable_gradient_checkpointing:
                actor_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
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
            from verl.utils.torch_functional import get_constant_schedule_with_warmup
            actor_optimizer = optim.AdamW(actor_module_fsdp.parameters(),
                                          lr=optim_config.lr,
                                          betas=optim_config.get('betas', (0.9, 0.999)),
                                          weight_decay=optim_config.get('weight_decay', 1e-2))

            total_steps = optim_config.get('total_training_steps', 0)
            num_warmup_steps_ratio = optim_config.get('lr_warmup_steps_ratio', 0.)
            num_warmup_steps = optim_config.get('lr_warmup_steps', None)
            if num_warmup_steps is not None and (isinstance(num_warmup_steps, bool) or int(num_warmup_steps) != num_warmup_steps):
                raise ValueError('Explicit warmup steps must be an integer')
            num_warmup_steps = int(num_warmup_steps_ratio * total_steps) if num_warmup_steps is None else int(num_warmup_steps)
            if num_warmup_steps < 0: raise ValueError('Warmup steps must be nonnegative')

            print(f'Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}')

            actor_lr_scheduler = get_constant_schedule_with_warmup(optimizer=actor_optimizer,
                                                                   num_warmup_steps=num_warmup_steps)
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
                                                               device_mesh=rollout_device_mesh)
            log_gpu_memory_usage('After building sharding manager', logger=None)

        return rollout, rollout_sharding_manager

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        from verl.workers.actor import DataParallelPPOActor
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get('external_lib', None))

        from omegaconf import OmegaConf
        override_model_config = OmegaConf.to_container(self.config.model.get('override_config', OmegaConf.create()))

        use_remove_padding = self.config.model.get('use_remove_padding', False)

        if self._is_actor or self._is_rollout:
            # we need the model for actor and rollout
            if self._is_actor:
                optim_config = self.config.actor.optim
                fsdp_config = self.config.actor.fsdp_config
            else:
                optim_config = None
                fsdp_config = OmegaConf.create()
            self.actor_module_fsdp, self.actor_optimizer, self.actor_lr_scheduler, self.actor_model_config = self._build_model_optimizer(
                model_path=self.config.model.path,
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
            if self.config.actor.get('baseline_method') == 'opsd':
                from search_r1.baselines.actor import DistillationActor
                self.actor = DistillationActor(self.config.actor, self.actor_module_fsdp,
                                               self.actor_optimizer, self.tokenizer)

        if self._is_rollout:
            self.rollout, self.rollout_sharding_manager = self._build_rollout()

        if self._is_ref:
            self.ref_module_fsdp = self._build_model_optimizer(model_path=self.config.ref.get('model_path',self.config.model.path),
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

        torch.cuda.empty_cache()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def inspect_fresh_initialization(self, expected_model_path):
        return inspect_fresh_worker_state(self, expected_model_path)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def calibrate_policy_safety(self, data: DataProto):
        """Dedicated healthy-checkpoint observation; never calls update_policy."""
        data = data.to('cuda')
        if self._is_offload_param:
            load_fsdp_param_and_grad(self.actor_module_fsdp, device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)
        from search_r1.baselines.sampling_consistency import calibrate_actor
        calibrate_actor(self, data)
        if self._is_offload_param:
            offload_fsdp_param_and_grad(self.actor_module_fsdp, offload_grad=self._is_offload_grad)
        torch.cuda.empty_cache()
        return DataProto(meta_info={'calibration_complete': True})

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data: DataProto):
        manager = getattr(self, 'rollout_sharding_manager', None)
        if manager is not None and getattr(manager, 'weights_resident', False):
            manager.release_model_weights()
        data = data.to('cuda')

        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.actor_module_fsdp,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=torch.cuda.current_device())

        data.batch = data.batch.cuda()

        log_gpu_memory_usage('Before update policy', logger=logger)

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            # Preserve actual counters before any startup refusal or audit IO.
            lr_before_outer = self.actor_lr_scheduler.get_last_lr()[0]
            scheduler_before_outer = self.actor_lr_scheduler.last_epoch
            self.actor._actual_committed_optimizer_calls = 0
            from search_r1.baselines.policy_safety_runtime import PolicySafetyRefusal
            from search_r1.baselines.partial_worker_state import prepare_partial_worker_context, persist_matched_partial_worker
            partial = None
            completed = None
            normal_guard = None
            from search_r1.baselines.tree_guard_continue import round_progress_enabled
            round_progress = round_progress_enabled(self.config.actor)
            try:
                if self.config.actor.get('fresh200_protocol') == 'tree_fresh200_five_leaf_v1':
                    prepare_partial_worker_context(self, data)
                with Timer(name='update_policy', logger=None) as timer:
                    metrics = self.actor.update_policy(data=data)
                if (getattr(self.actor, '_fresh_guard_refusal', None) is not None
                        and self.actor._actual_committed_optimizer_calls == 0 and not round_progress):
                    raise self.actor._fresh_guard_refusal
                if getattr(self.actor, '_fresh_guard_refusal', None) is not None and round_progress:
                    normal_guard = compact_normal_guard_after_actor(self, self.actor._fresh_guard_refusal)
                    metrics['actor/normal_guard_round_consumed'] = 1
                if round_progress:
                    metrics['actor/guard_round_progress_enabled'] = 1
                    metrics['actor/zero_write_sampling_round'] = int(self.actor._actual_committed_optimizer_calls == 0)
                delta_time = timer.last
                global_num_tokens = data.meta_info['global_token_num']
                estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
                metrics['mfu/actor'] = estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size

                if self.config.actor.get('fresh200_protocol') == 'tree_fresh200_five_leaf_v1':
                    self._fresh_direct_record = fresh200_scheduler_after_actor(self, data, metrics, scheduler_before_outer, lr_before_outer)
                elif metrics.get('actor/optimizer_steps', 0) > 0:
                    self.actor_lr_scheduler.step()
                lr = self.actor_lr_scheduler.get_last_lr()[0]
                metrics['actor/lr'] = lr
                metrics['actor/lr_before_outer'] = lr_before_outer
                metrics['actor/scheduler_step_before_outer'] = scheduler_before_outer
                metrics['actor/scheduler_step_after_outer'] = self.actor_lr_scheduler.last_epoch
                if getattr(self.actor, '_fresh_guard_refusal', None) is not None and not round_progress:
                    partial = persist_matched_partial_worker(self, data, self.actor._fresh_guard_refusal, scheduler_before_outer)
                from search_r1.baselines.tree_guard_continue import enabled as continue_enabled
                if continue_enabled(self.config.actor):
                    from search_r1.baselines.partial_worker_state import completed_actor_result
                    completed = completed_actor_result(self, data, scheduler_before_outer)

            except PolicySafetyRefusal as refusal:
                if (refusal.matched_all_ranks is not True
                        or self.config.actor.get('fresh200_protocol') != 'tree_fresh200_five_leaf_v1'):
                    raise
                if partial is None:
                    partial = persist_matched_partial_worker(self, data, refusal, scheduler_before_outer)
                metrics = {'actor/optimizer_steps': self.actor._actual_committed_optimizer_calls,
                           'actor/scheduler_step_before_outer': scheduler_before_outer,
                           'actor/scheduler_step_after_outer': self.actor_lr_scheduler.last_epoch,
                           'actor/partial_outer_refused': 1}

            log_gpu_memory_usage('After update policy', logger=logger)

            if partial is not None:
                from search_r1.baselines.partial_worker_state import reset_saved_guard_gradients
                partial = reset_saved_guard_gradients(self, partial)

            # TODO: here, we should return all metrics
            output = DataProto(meta_info={'metrics': metrics,
                **({'tree_partial_state': partial} if partial is not None else {}),
                **({'tree_fresh_actor_result': completed} if completed is not None else {}),
                **({'tree_normal_guard': normal_guard} if normal_guard is not None else {})})

            output = self.ulysses_sharding_manager.postprocess_data(data=output)
            output = output.to('cpu')

        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.actor_module_fsdp, offload_grad=self._is_offload_grad)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)
        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def release_rollout_policy_cache(self):
        manager = getattr(self, 'rollout_sharding_manager', None)
        if manager is not None and getattr(manager, 'weights_resident', False):
            manager.release_model_weights()

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_log_prob(self, data: DataProto) -> DataProto:
        """mostly copying from generate_sequences"""
        manager = getattr(self, 'rollout_sharding_manager', None)
        if manager is not None and getattr(manager, 'weights_resident', False):
            manager.release_model_weights()
        data = data.to('cuda')

        assert self._is_rollout
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.actor_module_fsdp,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)

        data.batch = data.batch.cuda()
        meta_info = {'eos_token_id': self.tokenizer.eos_token_id, 'pad_token_id': self.tokenizer.pad_token_id}
        data.meta_info.update(meta_info)

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            old_log_probs = self.actor.compute_log_prob(data=data)
            output = DataProto.from_dict(tensors={'old_log_probs': old_log_probs})
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
    def generate_sequences(self, prompts: DataProto):
        prompts = prompts.to('cuda')
        # set to False if it is validation
        recompute_log_prob = prompts.meta_info.get('recompute_log_prob', True)
        # Native recursive trajectories retain emitted behavior-mu directly.
        # The trainer scores the assembled trajectory once; a per-action actor
        # pass is only needed when the sampler/actor diagnostic is enabled.
        if (self.config.rollout.get('tree_raw_emission', False)
                and 'sampling_consistency' in self.config.rollout
                and not self.config.rollout.sampling_consistency.enabled):
            recompute_log_prob = False
        if 'sampling_consistency' in self.config.rollout:
            from search_r1.baselines.sampling_consistency import generation_preflight
            generation_preflight(self, prompts)

        assert self._is_rollout
        from search_r1.baselines.tree_guard_continue import execution_profile_bound
        cache_enabled = (execution_profile_bound()
                         and self.config.rollout.get('tree_raw_emission', False)
                         and not recompute_log_prob and prompts.meta_info.get('do_sample', True))
        cache_version = (self.actor._fresh_total_committed_optimizer_calls,
                         self.actor_lr_scheduler.last_epoch) if cache_enabled else None
        manager = self.rollout_sharding_manager
        if hasattr(manager, 'prepare_policy_cache'):
            manager.prepare_policy_cache(cache_enabled, cache_version)
        elif cache_enabled:
            raise RuntimeError('Native recursive policy cache requires FSDP-vLLM manager')
        actor_loaded = self._is_offload_param and not getattr(manager, 'weights_resident', False)
        if actor_loaded:
            load_fsdp_param_and_grad(module=self.actor_module_fsdp,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)

        prompts.batch = prompts.batch.cuda()
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

        if ('sampling_consistency' in self.config.rollout
                and self.config.rollout.sampling_consistency.enabled
                and prompts.meta_info.get('do_sample', True)):
            from search_r1.baselines.sampling_consistency import record_raw_turn
            record_raw_turn(self, output)

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
        assert self._is_ref

        data = data.to('cuda')

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
    def save_checkpoint(self, local_path, hdfs_path=None):
        manager = getattr(self, 'rollout_sharding_manager', None)
        if manager is not None and getattr(manager, 'weights_resident', False):
            manager.release_model_weights()
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


    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_recovery_training_state(self):
        from search_r1.baselines.training_state import recovery_manifest,verified_state_file,digest,restore_worker
        manifest=recovery_manifest()
        records=[r for r in manifest['worker_states'] if r['identity']['rank']==self.rank]
        if len(records)!=1:raise ValueError('Recovery rank coverage invalid')
        record=records[0]
        if record['identity']['world_size']!=self.world_size:
            raise ValueError('Recovery world/shard identity changed')
        saved=verified_state_file(record)
        if digest(saved)!=record['logical_state_sha256']:raise ValueError('Parent worker logical hash changed')
        if saved['identity']!=record['identity']:raise ValueError('Parent worker identity changed')
        expected_steps=int(os.environ['TREE_RECOVERY_PARENT_OPTIMIZER_STEPS'])
        if not saved['optimizer']['state'] or any(int(v['step'])!=expected_steps for v in saved['optimizer']['state'].values()):
            raise ValueError('Parent Adam step mismatch')
        # A fresh FSDP wrapper has not initialized its internal local shard yet.
        # Initialize on the reserved device, then return to the saved offload
        # layout before copying actor and Adam state. No dummy optimizer step.
        from search_r1.baselines.training_state import prepare_cold_fsdp_layout,worker_snapshot
        prepare_cold_fsdp_layout(self.actor_module_fsdp,self._is_offload_param,self._is_offload_grad)
        restore_worker(self.actor_module_fsdp,self.actor_optimizer,self.actor_lr_scheduler,
                       saved,record['identity'],cold=True)
        if digest(worker_snapshot(self.actor_module_fsdp,self.actor_optimizer,self.actor_lr_scheduler,
                                  record['identity'],cuda=True))!=record['logical_state_sha256']:
            raise ValueError('Cold restored worker state differs from parent snapshot')
        return dict(rank=self.rank,parent_identity=record['identity'],scheduler=self.actor_lr_scheduler.last_epoch,
            adam_steps=sorted(set(int(v['step']) for v in self.actor_optimizer.state.values())),
            parent_sha256=record['sha256'],scope='Cold state migration; not executor/vLLM equivalence')

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_current_chain_training_state(self):
        from search_r1.baselines.tree_current_chain_resume import load_worker
        return load_worker(self)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_training_state(self, local_path, identity, verify=False):
        if not self._is_actor or not self.config.actor.get('training_state_enabled', False):
            raise ValueError('Complete actor training state is not enabled')
        from search_r1.baselines.training_state import snapshot_worker_to_file
        identity = dict(identity, rank=self.rank, world_size=self.world_size,
                        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
        if 'guard_continue_accounting' in identity:
            expected = identity['guard_continue_accounting']['total_optimizer_calls_per_rank'][str(self.rank)]
            if (self.actor._fresh_total_committed_optimizer_calls != expected
                    or self.actor_lr_scheduler.last_epoch != identity['completed_step']):
                raise ValueError('Checkpoint actual cumulative optimizer/scheduler accounting differs')
        path = os.path.join(local_path, f'rank-{self.rank:02d}.pt')
        # At this boundary actor params/optimizer already occupy their configured
        # offload devices. Restore into that same layout without reinitializing.
        receipt = snapshot_worker_to_file(self.actor_module_fsdp, self.actor_optimizer,
                                       self.actor_lr_scheduler, path, identity,
                                       verify=verify, cuda=True)
        if 'guard_continue_accounting' in identity:
            receipt['actual_total_optimizer_calls'] = self.actor._fresh_total_committed_optimizer_calls
            receipt['observed_adam_steps'] = sorted(set(int(value['step']) for value in self.actor_optimizer.state.values()))
        return receipt


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
                                                                            attn_implementation='flash_attention_2',
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
                                                                            attn_implementation='flash_attention_2',
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
