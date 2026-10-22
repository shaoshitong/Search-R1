"""Independent native4 CUDA/NCCL health test; no models or training are loaded."""
import json
import os
from pathlib import Path
import sys
import time


def main():
    import torch
    import torch.distributed as dist
    rank = int(os.environ['LOCAL_RANK'])
    root = Path(sys.argv[1])
    report = dict(rank=rank, slurm_job_id=os.environ['SLURM_JOB_ID'],
                  hostname=os.uname().nodename, started_unix=time.time(),
                  cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  model_loaded=False, training_updates=0, status='FAILED')
    try:
        report['cuda_device_count'] = torch.cuda.device_count()
        assert report['cuda_device_count'] == 4
        torch.cuda.set_device(rank)
        props = torch.cuda.get_device_properties(rank)
        report.update(device_name=props.name, compute_capability=[props.major, props.minor],
                      actual_cuda_uuid=str(getattr(props, 'uuid', 'unavailable')))
        assert props.major == 9
        dist.init_process_group('nccl')
        x = torch.ones((128, 128), device=rank, dtype=torch.bfloat16, requires_grad=True)
        y = x @ torch.ones_like(x)
        y.float().mean().backward()
        assert torch.equal(y, torch.full_like(y, 128))
        assert torch.equal(x.grad, torch.full_like(x, 1 / 128))
        value = torch.tensor([rank + 1.0], device=rank)
        dist.all_reduce(value)
        torch.cuda.synchronize(rank)
        assert value.item() == 10.0
        report.update(status='NATIVE4_CUDA_NCCL_HEALTH_PASS',
                      actual_forward_backward=True, actual_nccl_all_reduce=10.0)
        dist.barrier()
    except BaseException as error:
        report.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report['finished_unix'] = time.time()
        (root / ('rank-%d.json' % rank)).write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report), flush=True)
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
