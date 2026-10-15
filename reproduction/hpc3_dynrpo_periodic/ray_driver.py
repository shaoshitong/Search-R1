"""CPU Ray entry that binds the admitted probe GPU before framework imports."""
import ray


@ray.remote(max_retries=0)
def periodic_main_task(config):
    from search_g1.qprobe_driver_gpu import bind_native_driver_gpu
    bind_native_driver_gpu(config)
    from verl.trainer.main_ppo_eco import run_main_task
    return run_main_task(config)


@ray.remote
def driver_import_smoke(expected_uuid):
    """Actual CPU-task CUDA/import check; no training claim or optimizer."""
    import os
    import torch
    from search_g1.qprobe_driver_gpu import install_driver_visibility
    from reproduction.common8.native_allocation import cuda_uuids
    assert ray.get_gpu_ids() == []
    incoming=os.environ.get('CUDA_VISIBLE_DEVICES')
    assert incoming in (None,'')
    proof=install_driver_visibility(dict(gpu_uuid=expected_uuid),environ=os.environ,
        cuda_initialized=torch.cuda.is_initialized,cuda_device_count=torch.cuda.device_count,
        cuda_uuids=cuda_uuids)
    from verl.trainer.main_ppo_eco import run_main_task
    assert callable(run_main_task) and cuda_uuids(1)==[expected_uuid]
    from transformers import AutoTokenizer
    from reproduction.hpc3_dynrpo_periodic.common import MODEL
    tokenizer=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True)
    assert os.environ.get('TOKENIZERS_PARALLELISM')=='false'
    assert os.environ.get('RAYON_NUM_THREADS')=='1'
    encoded=tokenizer(['Validate the actual tokenizer worker.']*64,padding=True,return_tensors='pt')
    assert encoded['input_ids'].shape[0]==64
    value=torch.zeros(1,device='cuda:0');torch.cuda.synchronize()
    assert value.item()==0
    return dict(proof,framework_import_after_binding=True,actual_cuda_compute=True,
                optimizer_updates=0,ray_gpu_ids=ray.get_gpu_ids(),actual_tokenizer_batch=64,
                tokenizer_parallelism=os.environ['TOKENIZERS_PARALLELISM'])
