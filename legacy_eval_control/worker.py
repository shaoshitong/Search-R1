"""Ray imports this unique package before resolving the pinned old workers."""
import os
from pathlib import Path
from legacy_eval_control import activate


def make_worker_class():
    activate()
    import torch
    import vllm,transformers
    if vllm.__version__!='0.6.3':raise ValueError('LEGACY_REQUIRES_VLLM_0_6_3')
    import torch.distributed as dist
    from omegaconf import OmegaConf
    from verl.workers.fsdp_workers import ActorRolloutRefWorker
    from verl.single_controller.base.decorator import register, Dispatch
    from reproduction.legacy_evaluation.contract import (PROTOCOL_PATH,digest,fp32_loader_options)

    class LegacyRolloutWorker(ActorRolloutRefWorker):
        def __init__(self,config,role):
            activate()
            if os.environ['SEARCH_G1_PROFILE']!='legacy_evaluation' or role!='actor_rollout':
                raise ValueError('LEGACY_WORKER_ROLE_CHANGED')
            super().__init__(config,role='rollout')

        def _build_model_optimizer(self,*args,**kwargs):
            if args:
                raise ValueError('LEGACY_LOADER_REQUIRES_EXPLICIT_KEYWORD_BINDING')
            options=fp32_loader_options(kwargs.get('fsdp_config'))
            options['fsdp_config']=OmegaConf.create(options['fsdp_config'])
            kwargs.update(options)
            return super()._build_model_optimizer(**kwargs)

        @register(dispatch_mode=Dispatch.ONE_TO_ALL)
        def init_model(self):
            activate()
            super().init_model()
            if self._is_actor or not self._is_rollout or self.actor_optimizer is not None or self.actor_lr_scheduler is not None:
                raise ValueError('LEGACY_WORKER_CONSTRUCTED_TRAINING_STATE')
            import json
            from transformers import GenerationConfig
            from reproduction.common8.sampling_capture import generation_source,loaded_generation
            from verl.third_party.vllm.vllm_v_0_6_3.llm_engine_sp import _load_generation_config_dict
            from reproduction.common8.native_allocation import cuda_uuids
            binding=json.loads(Path(os.environ['SEARCH_G1_LEGACY_BINDING']).read_text())
            engine=self.rollout.inference_engine.llm_engine
            provenance=dict(profile='common8',run_id=binding['run_id'],source_commit=binding['legacy_source_commit'],
                source_sha256=binding['legacy_source_sha256'],initial_model=binding['checkpoint'],
                initial_model_sha256={'generation_config.json':binding['generation_config_sha256']},
                execution_backend='slurm',slurm_job_id=os.environ['SLURM_JOB_ID'])
            original,source=generation_source(engine.model_config,_load_generation_config_dict,run=provenance)
            engine._common8_generation_provenance=loaded_generation(source,engine.generation_config_fields,
                GenerationConfig.from_dict(original).to_diff_dict())
            model=engine.model_executor.worker.model_runner.model
            hf=list(self.actor_module_fsdp.parameters());vp=list(model.parameters())
            if not hf or not vp:
                raise ValueError('LEGACY_PARAMETER_READBACK_EMPTY')
            scheduler=engine.scheduler_config
            return dict(rank=dist.get_rank(),world_size=dist.get_world_size(),
                cuda_uuid=cuda_uuids(count=torch.cuda.device_count())[torch.cuda.current_device()],
                run_id=binding['run_id'],checkpoint_step=binding['checkpoint_step'],checkpoint=binding['checkpoint'],checkpoint_manifest_sha256=binding['checkpoint_manifest_sha256'],
                generation_config_sha256=binding['generation_config_sha256'],recipe_sha256=digest(PROTOCOL_PATH),
                hf_parameter_dtypes=sorted({str(p.dtype) for p in hf}),hf_parameter_numel=sum(p.numel() for p in hf),
                vllm_parameter_dtypes=sorted({str(p.dtype) for p in vp}),vllm_parameter_numel=sum(p.numel() for p in vp),
                actual_vllm_config_dtype=str(engine.model_config.dtype),actual_vllm_version=vllm.__version__,
                actual_torch_version=torch.__version__,actual_transformers_version=transformers.__version__,
                actual_engine_max_model_len=engine.model_config.max_model_len,
                actual_engine_max_num_seqs=scheduler.max_num_seqs,
                actual_engine_max_num_batched_tokens=scheduler.max_num_batched_tokens,
                optimizer_constructed=False,optimizer_steps=0,actual_model_loaded=True,
                loaded_old_source_root=str(activate()))

        @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
        def generate_sequences(self,prompts):
            if prompts.meta_info.get('do_sample') is not False or prompts.meta_info.get('recompute_log_prob') is not False:
                raise ValueError('LEGACY_EVALUATION_GREEDY_REQUIRED')
            before=os.environ['SEARCH_G1_PROFILE'];os.environ['SEARCH_G1_PROFILE']='common8'
            try:return super().generate_sequences(prompts)
            finally:os.environ['SEARCH_G1_PROFILE']=before

    return LegacyRolloutWorker
