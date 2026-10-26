"""通过 su 以 root 运行；只在指定空卡启动独立、限时的 systemd 服务。"""
import argparse
import csv
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--python', required=True)
    p.add_argument('--gpus', required=True)
    p.add_argument('--unit', required=True)
    p.add_argument('--sha256', required=True)
    a = p.parse_args()
    assert os.geteuid() == 0, '必须通过 su 以 root 启动'
    root = Path(__file__).resolve().parent
    script = root / 'train.py'
    assert hashlib.sha256(script.read_bytes()).hexdigest() == a.sha256
    wanted = [int(x) for x in a.gpus.split(',')]
    assert wanted in ([4, 5], [4, 5, 6, 7])
    assert a.unit.startswith('resnet50-noise-20261026-')
    state = subprocess.run(['systemctl', 'show', a.unit, '-p', 'LoadState', '--value'],
                           stdout=subprocess.PIPE, universal_newlines=True).stdout.strip()
    if state != 'not-found':
        raise RuntimeError('服务已存在，拒绝重复启动：' + a.unit)
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,utilization.gpu,memory.used',
                                   '--format=csv,noheader,nounits'], universal_newlines=True)
    rows = {int(r[0]): [v.strip() for v in r] for r in csv.reader(io.StringIO(raw))}
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                    '--format=csv,noheader,nounits'], universal_newlines=True)
    occupied = {r[0].strip() for r in csv.reader(io.StringIO(apps)) if r}
    for i in wanted:
        row = rows[i]
        if row[1] in occupied or int(row[2]) != 0 or int(row[3]) > 32:
            raise RuntimeError('指定 GPU 当前非空闲：' + str(i))
    uuids = [rows[i][1] for i in wanted]
    output = root / 'progress'
    output.mkdir(exist_ok=True)
    command = [
        'systemd-run', '--unit=' + a.unit, '--service-type=exec',
        '--property=RuntimeMaxSec=86400', '--property=TimeoutStopSec=15',
        '--property=KillMode=control-group', '--property=Restart=no',
        '--property=WorkingDirectory=' + str(root), '--property=UMask=0022',
        '--property=StandardOutput=append:' + str(root / 'train.log'),
        '--property=StandardError=append:' + str(root / 'train.log'),
        '--setenv=CUDA_VISIBLE_DEVICES=' + ','.join(uuids),
        '--setenv=CUDA_DEVICE_ORDER=PCI_BUS_ID', '--setenv=PYTHONUNBUFFERED=1',
        '--setenv=OMP_NUM_THREADS=4', '--setenv=MKL_NUM_THREADS=4',
        '--setenv=NCCL_P2P_DISABLE=1',
        '--setenv=TORCH_NCCL_ASYNC_ERROR_HANDLING=1',
        a.python, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
        '--nproc_per_node=' + str(len(wanted)), str(script),
        '--batch-size=1024', '--micro-batch=64', '--seconds=86400', '--output=' + str(output),
    ]
    subprocess.run(command, check=True)
    receipt = dict(unit=a.unit + '.service', gpu_indices=wanted, gpu_uuids=uuids,
                   python=a.python, batch_size_per_node=1024, micro_batch_per_gpu=64,
                   duration_seconds=86400, train_sha256=a.sha256,
                   launched_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   root=str(root))
    (root / 'launch.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
