"""Independent native4 CUDA/NCCL health test; no models or training are loaded."""
import json
import os
from pathlib import Path
import shlex
import subprocess
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
        device = torch.device('cuda', rank)
        x = torch.ones((128, 128), device=device, dtype=torch.bfloat16, requires_grad=True)
        y = x @ torch.ones_like(x)
        y.float().mean().backward()
        assert torch.equal(y, torch.full_like(y, 128))
        assert torch.equal(x.grad, torch.full_like(x, 1 / 128))
        value = torch.tensor([rank + 1.0], device=device)
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


def submit():
    root = Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fastpath-probe-20261022/diagnostics/node13-health-after749812-r1-20261023')
    root.mkdir(parents=True, exist_ok=False)
    python = '/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fa3-leanmetrics-r1-20261023/env/bin/python'
    script = root / 'job.sh'
    script.write_text('#!/bin/bash\nset -euo pipefail\n'
                      'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1\nexec ' +
                      shlex.join([python, '-B', '-m', 'torch.distributed.run', '--standalone',
                                  '--nproc_per_node=4', str(Path(__file__).resolve()), str(root)]) + '\n')
    command = ['sbatch', '--parsable', '--partition=emergency_acd', '--qos=formal-user',
               '--account=jxu957', '--nodelist=ACD1-13', '--nodes=1', '--ntasks=1',
               '--gres=gpu:4', '--cpus-per-task=8', '--mem=32G', '--time=00:02:00',
               '--export=NONE', '--job-name=easy7b-node13-health',
               '--output='+str(root/'slurm.log'), '--error='+str(root/'slurm.log'), str(script)]
    claim = dict(epoch=time.time(), source_file=str(Path(__file__).resolve()), command=command,
                 model_loaded=False, training_updates=0, formal_job750029_modified=False)
    (root/'claim.json').write_text(json.dumps(claim, indent=2)+'\n')
    result = subprocess.run(command, capture_output=True, text=True, timeout=20)
    value = dict(claim, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    if result.returncode == 0:
        value['job'] = result.stdout.strip().split(';')[0]
        (root/'job.id').write_text(value['job']+'\n')
    (root/'submission.json').write_text(json.dumps(value, indent=2)+'\n')
    print(json.dumps(value), flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'submit':
        submit()
    else:
        main()
