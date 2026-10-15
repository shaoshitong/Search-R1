"""Rollout-only FP32 worker over current async infrastructure, no old tree swap."""
import contextlib
import json
import os
from pathlib import Path

PROFILE = 'compatible_async_evaluation'
RECIPE_ID = 'old-context-fp32-async'


def fp32_loader_options(fsdp_config):
    options = dict(fsdp_config or {})
    options['model_dtype'] = 'fp32'
    options['mixed_precision'] = dict(param_dtype='fp32', reduce_dtype='fp32', buffer_dtype='fp32')
    return dict(fsdp_config=options, optim_config=None, use_remove_padding=False,
                enable_gradient_checkpointing=False)


@contextlib.contextmanager
def phase_profile():
    """Worker-local compatibility for provenance-aware common8 phase methods."""
    before = os.environ.get('SEARCH_G1_PROFILE')
    if before != PROFILE:
        raise ValueError('Evaluation worker profile changed')
    os.environ['SEARCH_G1_PROFILE'] = 'common8'
    try:
        yield
    finally:
        os.environ['SEARCH_G1_PROFILE'] = before


def make_worker_class(binding_env='SEARCH_G1_COMPAT_EVAL_BINDING'):
    import torch
    import vllm, transformers
    import torch.distributed as dist
    from omegaconf import OmegaConf
    from transformers import GenerationConfig
    from verl.workers.fsdp_workers import ActorRolloutRefWorker
    from verl.single_controller.base.decorator import register, Dispatch

    class CompatibleAsyncEvaluationWorker(ActorRolloutRefWorker):
        def __init__(self, config, role):
            if os.environ.get('SEARCH_G1_PROFILE') != PROFILE or role != 'actor_rollout':
                raise ValueError('Compatible evaluation worker role changed')
            if config.rollout.dtype != 'float32' or not config.rollout.get('async_phase_enabled', False):
                raise ValueError('FP32 async evaluation rollout must be explicit')
            super().__init__(config, role='rollout')

        def _build_model_optimizer(self, *args, **kwargs):
            if args:
                raise ValueError('Evaluation loader requires explicit keyword arguments')
            options = fp32_loader_options(kwargs.get('fsdp_config'))
            options['fsdp_config'] = OmegaConf.create(options['fsdp_config'])
            kwargs.update(options)
            return super()._build_model_optimizer(**kwargs)

        @register(dispatch_mode=Dispatch.ONE_TO_ALL)
        def init_model(self):
            super().init_model()
            if self._is_actor or not self._is_rollout or self.actor_optimizer is not None or self.actor_lr_scheduler is not None:
                raise ValueError('Evaluation constructed training optimizer/scheduler')
            binding = json.loads(Path(os.environ[binding_env]).read_text())
            if binding.get('recipe_id') != RECIPE_ID:
                raise ValueError('Wrong compatible evaluation recipe')
            from reproduction.common8.sampling_capture import generation_source, loaded_generation
            from verl.third_party.vllm.vllm_v_0_6_3.llm_engine_sp import _load_generation_config_dict
            from reproduction.common8.native_allocation import cuda_uuids
            engine = self.rollout.inference_engine.llm_engine
            provenance_run = dict(profile='common8', run_id=binding['run_id'],
                source_commit=binding['source_commit'], source_sha256=binding['source_sha256'],
                initial_model=binding['checkpoint'],
                initial_model_sha256={'generation_config.json':binding['generation_config_sha256']},
                execution_backend='slurm', slurm_job_id=os.environ['SLURM_JOB_ID'])
            original, source = generation_source(engine.model_config, _load_generation_config_dict, run=provenance_run)
            engine._common8_generation_provenance = loaded_generation(source,
                engine.generation_config_fields, GenerationConfig.from_dict(original).to_diff_dict())
            hf = list(self.actor_module_fsdp.parameters())
            actual = engine.model_executor.worker.model_runner.model
            native = list(actual.parameters())
            if not hf or not native or any(p.dtype != torch.float32 for p in hf+native):
                raise ValueError('Actual HF/vLLM parameters are not FP32')
            if engine.model_config.dtype != torch.float32 or engine.model_config.max_model_len != 4596:
                raise ValueError('Actual vLLM dtype/context changed')
            return dict(rank=dist.get_rank(), world_size=dist.get_world_size(),
                cuda_uuid=cuda_uuids(count=torch.cuda.device_count())[torch.cuda.current_device()],
                checkpoint=binding['checkpoint'], checkpoint_step=binding['checkpoint_step'],
                checkpoint_manifest_sha256=binding['checkpoint_manifest_sha256'],
                generation_config_sha256=binding['generation_config_sha256'],
                recipe_id=RECIPE_ID, recipe_sha256=binding['recipe_sha256'],
                hf_parameter_dtypes=sorted({str(p.dtype) for p in hf}),
                vllm_parameter_dtypes=sorted({str(p.dtype) for p in native}),
                hf_parameter_numel=sum(p.numel() for p in hf),
                vllm_parameter_numel=sum(p.numel() for p in native),
                actual_vllm_config_dtype=str(engine.model_config.dtype),
                actual_engine_max_model_len=engine.model_config.max_model_len,
                optimizer_constructed=False, optimizer_steps=0, actual_model_loaded=True,
                async_phase_enabled=True, actual_vllm_version=vllm.__version__,
                actual_torch_version=torch.__version__, actual_transformers_version=transformers.__version__,
                actual_engine_max_num_seqs=engine.scheduler_config.max_num_seqs,
                actual_engine_max_num_batched_tokens=engine.scheduler_config.max_num_batched_tokens)

        @register(dispatch_mode=Dispatch.ONE_TO_ALL)
        def begin_generation_phase(self, snapshot_meta):
            identity = snapshot_meta.get('behavior_identity', {})
            if identity.get('validation') is not True:
                raise ValueError('Evaluation phase cannot sample training trajectories')
            with phase_profile():
                return super().begin_generation_phase(snapshot_meta)

        @register(dispatch_mode=Dispatch.DP_COMPUTE, blocking=False)
        @torch.no_grad()
        def advance_generation_phase(self, handle, new_requests, max_steps=8):
            for request in new_requests:
                meta = request['meta_info']
                if meta.get('do_sample') is not False or meta.get('recompute_log_prob') is not False:
                    raise ValueError('Evaluation phase requires greedy sampling without HF recompute')
            with phase_profile():
                return super().advance_generation_phase(handle, new_requests, max_steps=max_steps)

        @register(dispatch_mode=Dispatch.ONE_TO_ALL)
        def end_generation_phase(self, handle, abort=False):
            with phase_profile():
                return super().end_generation_phase(handle, abort=abort)

    return CompatibleAsyncEvaluationWorker
