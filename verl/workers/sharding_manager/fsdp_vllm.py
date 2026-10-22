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

import os
import logging
import torch
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import ShardingStrategy, ShardedStateDictConfig, StateDictType, FullStateDictConfig
from torch.distributed.device_mesh import DeviceMesh

from verl.third_party.vllm import LLM
from verl.third_party.vllm import parallel_state as vllm_ps
from verl import DataProto
from verl.utils.torch_functional import (broadcast_dict_tensor, allgather_dict_tensors)
from verl.utils.debug import log_gpu_memory_usage

from .base import BaseShardingManager

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_PPO_LOGGING_LEVEL', 'WARN'))


class FSDPVLLMShardingManager(BaseShardingManager):

    def __init__(self,
                 module: FSDP,
                 inference_engine: LLM,
                 model_config,
                 full_params: bool = False,
                 device_mesh: DeviceMesh = None,
                 reuse_snapshot_weights: bool = False):
        if type(reuse_snapshot_weights) is not bool:
            raise ValueError('Snapshot weight reuse must be an explicit bool')
        self.module = module
        self.inference_engine = inference_engine
        self.model_config = model_config
        self.device_mesh = device_mesh
        self._phase_snapshot = None
        self._phase_setup_snapshot = None
        self._rng_swapped = False
        self.phase_weight_sync_count = 0
        self.reuse_snapshot_weights = reuse_snapshot_weights
        self._resident_weight_snapshot = None
        self._retain_phase_weights = False

        # Full params
        self.full_params = full_params
        if full_params:
            FSDP.set_state_dict_type(self.module,
                                     state_dict_type=StateDictType.FULL_STATE_DICT,
                                     state_dict_config=FullStateDictConfig())
        else:
            FSDP.set_state_dict_type(self.module,
                                     state_dict_type=StateDictType.SHARDED_STATE_DICT,
                                     state_dict_config=ShardedStateDictConfig())

        # Note that torch_random_states may be different on each dp rank
        self.torch_random_states = torch.cuda.get_rng_state()
        # get a random rng states
        if self.device_mesh is not None:
            gen_dp_rank = self.device_mesh['dp'].get_local_rank()
            torch.cuda.manual_seed(gen_dp_rank + 1000)  # make sure all tp ranks have the same random states
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)
        else:
            self.gen_random_states = None

    def __enter__(self):
        # The legacy context manager never trusts a previous sampling snapshot.
        self.release_snapshot_weights()
        return self._enter(sync_weights=True)

    def _enter(self, *, sync_weights, own_weights=False):
        if self._phase_snapshot is not None:
            raise RuntimeError('Per-request FSDP synchronization inside resident rollout phase')
        self._rng_swapped = False
        if sync_weights:
            log_gpu_memory_usage('Before state_dict() in sharding manager memory', logger=logger)
            params = self.module.state_dict()
            log_gpu_memory_usage('After state_dict() in sharding manager memory', logger=logger)
            # The normal loader copies into vLLM-owned storage. Never retain
            # the actor state dict or the uninitialized offload CPU buffers.
            load_format = 'hf' if self.full_params else 'dtensor'
            self.inference_engine.sync_model_weights(params, load_format=load_format)
            if own_weights:
                self._own_sampling_weights()
            log_gpu_memory_usage('After sync model weights in sharding manager', logger=logger)
            del params
            torch.cuda.empty_cache()
            log_gpu_memory_usage('After del state_dict and empty_cache in sharding manager', logger=logger)

        # TODO: offload FSDP model weights
        # self.module.cpu()
        # torch.cuda.empty_cache()
        # if torch.distributed.get_rank() == 0:
        # print(f'after model to cpu in sharding manager memory allocated: {torch.cuda.memory_allocated() / 1e9}GB, reserved: {torch.cuda.memory_reserved() / 1e9}GB')

        # important: need to manually set the random states of each tp to be identical.
        if self.device_mesh is not None:
            self.torch_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.gen_random_states)
            self._rng_swapped = True

    def _own_sampling_weights(self):
        """Retain only storage owned by the local vLLM model, never actor aliases.

        The native TP1 SPMD executor has one local model. Checking storage
        identity does not read tensor contents or copy non-aliased weights.
        Some legacy loaders alias sources; detach only those exact aliases.
        """
        model = self.inference_engine.llm_engine.model_executor.worker.model_runner.model
        if model is self.module:
            raise RuntimeError('Sampling snapshot cannot be the actor module')
        sources = list(self.module.parameters()) + list(self.module.buffers())
        def storage(value):
            local = value.to_local() if hasattr(value, 'to_local') else value
            return (str(local.device), local.untyped_storage().data_ptr()) if local.numel() else None
        actor_storage = {storage(value) for value in sources} - {None}
        checked = detached = 0
        for value in list(model.parameters()) + list(model.buffers()):
            checked += 1
            if storage(value) in actor_storage:
                value.data = value.detach().clone()
                detached += 1
            if storage(value) in actor_storage:
                raise RuntimeError('Sampling weight still aliases actor storage')
        self._last_weight_storage_proof = dict(checked_tensors=checked, detached_actor_aliases=detached,
                                               actor_storage_aliases_after=0)

    def __exit__(self, exc_type, exc_value, traceback):
        if not self._retain_phase_weights or exc_type is not None:
            log_gpu_memory_usage('Before vllm offload in sharding manager', logger=logger)
            self.inference_engine.offload_model_weights()
            self._resident_weight_snapshot = None
            log_gpu_memory_usage('After vllm offload in sharding manager', logger=logger)

        # self.module.to('cuda')
        # if torch.distributed.get_rank() == 0:
        #     print(f'after actor module to cuda in sharding manager memory allocated: {torch.cuda.memory_allocated() / 1e9}GB, reserved: {torch.cuda.memory_reserved() / 1e9}GB')

        self.module.train()

        # add empty cache after each compute
        torch.cuda.empty_cache()

        # restore random states
        if self.device_mesh is not None and self._rng_swapped:
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)
            self._rng_swapped = False

    def phase_needs_weight_sync(self, policy_snapshot_id, reuse=False):
        """Called before actor shards are loaded, identically on all ranks."""
        if type(reuse) is not bool or (reuse and not self.reuse_snapshot_weights):
            raise ValueError('Undeclared snapshot weight reuse')
        if self._phase_snapshot is not None or self._phase_setup_snapshot is not None:
            raise RuntimeError('Cannot inspect weights inside an active generation phase')
        hit = reuse and self._resident_weight_snapshot == policy_snapshot_id
        if not hit:
            self.release_snapshot_weights()
        return not hit

    def release_snapshot_weights(self):
        """Invalidate before actor work/restore, or after the last candidate chunk."""
        if self._phase_snapshot is not None:
            raise RuntimeError('Cannot release an active generation snapshot')
        if self._resident_weight_snapshot is not None:
            self.inference_engine.offload_model_weights()
            self._resident_weight_snapshot = None
            torch.cuda.empty_cache()

    def begin_phase(self, policy_snapshot_id, *, reuse=False, retain_weights=False):
        if (type(policy_snapshot_id) is not str or not policy_snapshot_id or
                self._phase_snapshot is not None or self._phase_setup_snapshot is not None):
            raise ValueError('Missing snapshot or overlapping resident rollout phase')
        if type(retain_weights) is not bool or (retain_weights and not reuse):
            raise ValueError('Retained weights require explicit snapshot reuse')
        sync_weights = self.phase_needs_weight_sync(policy_snapshot_id, reuse=True) if reuse else True
        self._phase_setup_snapshot = policy_snapshot_id
        self._retain_phase_weights = retain_weights
        if reuse:
            self._enter(sync_weights=sync_weights, own_weights=True)
            self._resident_weight_snapshot = policy_snapshot_id
        else:
            self.__enter__()
        self._phase_snapshot = policy_snapshot_id
        self.phase_weight_sync_count += int(sync_weights)

    def end_phase(self, policy_snapshot_id, abort=False):
        if self._phase_setup_snapshot != policy_snapshot_id:
            raise ValueError('Closing a different resident actor snapshot')
        try:
            if abort:
                self._retain_phase_weights = False
            self.__exit__(None, None, None)
        finally:
            self._phase_snapshot = None
            self._phase_setup_snapshot = None
            self._retain_phase_weights = False

    def preprocess_data(self, data: DataProto) -> DataProto:
        # TODO: Current impl doesn't consider FSDP with torch micro-dp
        data.batch = allgather_dict_tensors(data.batch.contiguous(),
                                            size=vllm_ps.get_tensor_model_parallel_world_size(),
                                            group=vllm_ps.get_tensor_model_parallel_group(),
                                            dim=0)

        return data

    def postprocess_data(self, data: DataProto) -> DataProto:
        # TODO: Current impl doesn't consider FSDP with torch micro-dp
        broadcast_dict_tensor(data.batch,
                              src=vllm_ps.get_tensor_model_parallel_src_rank(),
                              group=vllm_ps.get_tensor_model_parallel_group())
        dp_rank = torch.distributed.get_rank()
        dp_size = torch.distributed.get_world_size()  # not consider torch micro-dp
        tp_size = vllm_ps.get_tensor_model_parallel_world_size()
        if tp_size > 1:
            # TODO: shall we build a micro_dp group for vllm when integrating with vLLM?
            local_prompts = data.chunk(chunks=tp_size)
            data = local_prompts[dp_rank % tp_size]
        return data
