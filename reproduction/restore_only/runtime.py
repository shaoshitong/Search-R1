"""Actual eight-rank restoration, with no trainer.fit or sampling entry point."""
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace


def make_worker_class():
    from verl.workers.fsdp_workers import ActorRolloutRefWorker
    from verl.single_controller.base.decorator import register,Dispatch
    class RestoreOnlyWorker(ActorRolloutRefWorker):
        # Deliberately expose no usable update/sampling APIs in this acceptance.
        def update_actor(self,*args,**kwargs):raise RuntimeError('RESTORE_ONLY_NO_UPDATE')
        def generate_sequences(self,*args,**kwargs):raise RuntimeError('RESTORE_ONLY_NO_GENERATION')
        def begin_generation_phase(self,*args,**kwargs):raise RuntimeError('RESTORE_ONLY_NO_GENERATION')
        @register(dispatch_mode=Dispatch.ONE_TO_ALL)
        def acceptance_readback(self,receipt_path):
            import torch
            from search_g1.qprobe_checkpoint import read_source,boundary_worker_rng
            from search_g1.full_state_restore import equal_tree
            from search_g1.training_state import cpu_tree,worker_rng
            _,receipt,_,_,_=read_source(receipt_path)
            record=next(v for v in receipt['files'] if v['relative_path']==f'training_state/rank-{self.rank}.pt')
            saved=torch.load(record['path'],map_location='cpu',weights_only=False)
            equal_tree(cpu_tree(self.actor_optimizer.state_dict()),saved['optimizer'])
            equal_tree(cpu_tree(self.actor_lr_scheduler.state_dict()),saved['scheduler'])
            equal_tree(worker_rng(self),boundary_worker_rng(receipt,self.rank,saved))
            assert self.actor._optimizer_writes==saved['actual_optimizer_writes']
            assert self.actor._complete_outers==receipt['step']
            assert all(p.dtype==torch.float32 for p in self.actor_module_fsdp.parameters())
            assert all(v['exp_avg'].dtype==torch.float32 and v['exp_avg_sq'].dtype==torch.float32 for v in self.actor_optimizer.state_dict()['state'].values())
            return dict(rank=self.rank,actual_optimizer_writes=self.actor._optimizer_writes,
                scheduler_last_epoch=self.actor_lr_scheduler.last_epoch,
                final_actor_adam_scheduler_rng_equal=True,actual_fp32_master_and_adam_verified=True,actual_updates_executed=0)
    return RestoreOnlyWorker


def actual_loader(config,tokenizer):
    from torch.utils.data import DataLoader
    from verl.utils.dataset.rl_dataset import RLHFDataset,collate_fn
    dataset=RLHFDataset(parquet_files=config.data.train_files,tokenizer=tokenizer,
        prompt_key=config.data.prompt_key,max_prompt_length=config.data.max_prompt_length,
        filter_prompts=True,return_raw_chat=config.data.get('return_raw_chat',False),truncation='error')
    count=config.data.train_data_num
    if count is not None and count<=len(dataset.dataframe):dataset.dataframe=dataset.dataframe.sample(count,random_state=42)
    assert len(dataset)==169028
    return DataLoader(dataset,batch_size=config.data.train_batch_size,
        shuffle=config.data.shuffle_train_dataloader,drop_last=True,collate_fn=collate_fn)


def restore_live_driver(config,receipt_path,group,tokenizer):
    import torch
    from search_g1.qprobe_checkpoint import read_source,restore_driver
    from search_g1.full_state_restore import equal_tree
    from search_g1.qprobe_runtime import QProbeRuntime,QProbeCandidateStream
    from search_g1.training_state import driver_rng,compact_stream_state
    _,receipt,_,_,_=read_source(receipt_path)
    saved=torch.load(receipt['boundary_driver']['path'],map_location='cpu',weights_only=False)
    # Restore-only managers bind saved immutable version, never score or run RPC.
    driver=SimpleNamespace(reward_fn=SimpleNamespace(no_self=True),val_reward_fn=None)
    proof=restore_driver(driver,receipt_path)
    loader=actual_loader(config,tokenizer)
    stream=QProbeCandidateStream(loader,restored_state=driver._restored_candidate_state)
    equal_tree(compact_stream_state(stream.state()),compact_stream_state(saved['candidate_stream']))
    runtime=QProbeRuntime(config.question_probe,os.environ['SEARCH_G1_BASE'],tokenizer,worker_group=group)
    restored=runtime.restore(driver._restored_qprobe_state['path'],receipt['actor_snapshot_id'])
    from search_g1.qprobe_model import OnlineQProbe
    expected=OnlineQProbe.load(Path(receipt['question_probe_state_path']).parent/restored['head_file'],
        expected_actor_snapshot_id=receipt['actor_snapshot_id'],expected_feature_spec_sha=receipt['feature_spec_sha'])
    # This RPC executes head.state_dict on the actual allocated rank7 object;
    # load_state_dict's True acknowledgement is never treated as readback.
    actual=runtime.head.state_dict()
    equal_tree(actual,expected.state_dict())
    equal_tree(runtime.selection_rng.bit_generator.state,restored['selection_rng'])
    assert runtime.completed_attempts==set(restored['completed_attempts'])
    assert runtime.completed_outer==receipt['step']
    # Construction must not perturb the saved driver RNG. Reset is forbidden here.
    equal_tree(driver_rng(),saved['rng'])
    return dict(proof,actual_candidate_stream_materialized_equal=True,
        actual_probe_rank7_full_state_readback_equal=True,
        actual_probe_device=runtime.head.worker_receipt['actual_probe_device'],
        actual_probe_cuda_uuid=runtime.head.worker_receipt['gpu_uuid'],
        actual_probe_rank=runtime.head.worker_receipt['worker_rank'],
        actual_selection_rng_equal=True,actual_completed_attempts_equal=True)


def run_step(binding,stepbase,receipt_path):
    import ray
    from omegaconf import OmegaConf,open_dict
    from search_g1.qprobe_checkpoint import read_source
    from search_g1.full_state_restore import publish
    from reproduction.restore_only.contract import bind_step
    _,receipt,checkpoint,_,parent=read_source(receipt_path)
    run=bind_step(parent,receipt,receipt_path,stepbase)
    run['native_ray_execution_id']=__import__('hashlib').md5((binding['execution_id']+':'+str(receipt['step'])).encode()).hexdigest()
    publish(stepbase/'logs/run-manifest.json',run)
    # Saved reward probe metadata only; no reward inference/refit is performed.
    publish(stepbase/'artifacts/live/manifest.json',dict(run_id=run['run_id'],version=receipt['reward_probe_version'],round=receipt['reward_probe_round']))
    config=OmegaConf.load(binding['resolved_config'])
    with open_dict(config):
        config.actor_rollout_ref.model.actor_checkpoint_path=str(checkpoint)
        config.actor_rollout_ref.actor.optim.completed_updates=receipt['step']
        config.actor_rollout_ref.actor.optim.total_training_steps=201
        config.question_probe=run['question_probe']
    from verl.utils import hf_tokenizer
    tokenizer=hf_tokenizer(config.actor_rollout_ref.model.path)
    from verl.single_controller.ray import RayWorkerGroup,RayClassWithInitArgs,RayResourcePool
    pool=RayResourcePool(process_on_nodes=[8],use_gpu=True,max_colocate_count=1,name_prefix='restore_only')
    group=RayWorkerGroup(resource_pool=pool,ray_cls_with_init=RayClassWithInitArgs(
        cls=ray.remote(make_worker_class()),config=config.actor_rollout_ref,role='actor_rollout'))
    try:
        group.init_model()
        workers=group.restore_mixed_full_state(str(receipt_path))
        assert len(workers)==8 and sorted(v['rank'] for v in workers)==list(range(8))
        assert {v['actual_cuda_uuid'] for v in workers}==set(binding['physical_gpu_uuids'])
        assert all(v['actual_optimizer_writes']==receipt['attempt_progress']['cumulative_writes'] and
            v['scheduler_last_epoch']==receipt['step'] and v['actual_adam_scheduler_readback_equal'] and
            v['actual_rng_readback_equal'] for v in workers)
        driver=restore_live_driver(config,receipt_path,group,tokenizer)
        final=group.acceptance_readback(str(receipt_path))
        assert len(final)==8 and all(v['final_actor_adam_scheduler_rng_equal'] for v in final)
        result=dict(role='actual_full_state_restore_only_readback',run_id=run['run_id'],step=receipt['step'],
            actual_cuda_full_state_restore_accepted=True,workers=workers,final_worker_readbacks=final,
            driver=driver,training_updates_executed=0,generation_calls=0,next_outer_executed=False)
        publish(stepbase/'logs/restore-only-result.json',result)
        return result
    finally:
        # The coordinator waits for actual process/GPU release before next point.
        for worker in group.workers:ray.kill(worker,no_restart=True)
