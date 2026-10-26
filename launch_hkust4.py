"""HKUST4 专用入口：验收交接完成且四卡空闲后，启动 24 小时随机训练。"""
import argparse
import csv
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess

USER = 'shaoshitong'
SHARED = '/weka_3/shaoshitong/lse-sparse-repro-20261026/shared'
GATE = SHARED + '/WAN_ACCEPTED_GPU_HANDOFF.json'
EXPECTED = {
    2: 'GPU-f2fa94d1-ac5e-9497-7d90-d41d5057c234',
    3: 'GPU-3649579c-dac9-4462-06e1-0c4c03ce9fab',
    5: 'GPU-79de0d51-c2a2-897e-f76a-ec84f03b38e2',
    6: 'GPU-3a89b647-73fb-27bc-c8a8-e9f663b96d5f',
}


def user_read(path):
    # SSHFS 未开放 allow_other，按挂载用户读取；不修改挂载设置。
    return subprocess.check_output(['runuser', '-u', USER, '--', 'cat', path], text=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--python', required=True)
    p.add_argument('--unit', required=True)
    p.add_argument('--sha256', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    assert os.geteuid() == 0, '请通过 sudo 调用启动入口'
    assert socket.gethostname().split('.')[0] == 'xielab5', '目标机器不符'
    assert a.unit.startswith('resnet50-noise-20261026-hkust4-')
    assert a.output.startswith(SHARED + '/resnet50-')
    root = Path(__file__).resolve().parent
    script = root / 'train.py'
    assert hashlib.sha256(script.read_bytes()).hexdigest() == a.sha256
    gate_text = user_read(GATE)
    gate = json.loads(gate_text)
    if gate.get('status') != 'accepted':
        raise RuntimeError('Wan 尚未验收通过，禁止启动')
    state = subprocess.check_output(['systemctl', 'show', a.unit, '-p', 'LoadState', '--value'], text=True).strip()
    if state != 'not-found':
        raise RuntimeError('服务已存在，拒绝重复启动：' + a.unit)
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    rows = {int(r[0]): [x.strip() for x in r] for r in csv.reader(io.StringIO(raw))}
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True)
    occupied = {r[0].strip() for r in csv.reader(io.StringIO(apps)) if r}
    for i, uuid in EXPECTED.items():
        row = rows[i]
        if row[1] != uuid or uuid in occupied or int(row[2]) > 32 or int(row[3]) != 0:
            raise RuntimeError('指定卡 UUID 不符或非空闲：' + str(i))
    # 独立输出目录由挂载用户创建。训练也使用该用户，保留 SSHFS 的访问权限。
    subprocess.run(['runuser', '-u', USER, '--', 'mkdir', a.output], check=True)
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]
    train = [a.python, '-m', 'torch.distributed.run', '--nnodes=1', '--node-rank=0',
             '--rdzv-backend=static', '--master-addr=127.0.0.1', '--master-port=' + str(port),
             '--nproc_per_node=4', str(script), '--batch-size=2048', '--micro-batch=128',
             '--seconds=86400', '--output=' + a.output + '/progress']
    # 在用户进程中打开共享日志，避免 systemd 以 root 打开 SSHFS 文件时权限失败。
    wrapped = 'exec ' + shlex.join(train) + ' >> ' + shlex.quote(a.output + '/train.log') + ' 2>&1'
    command = ['systemd-run', '--unit=' + a.unit, '--service-type=exec',
               '--property=User=' + USER, '--property=RuntimeMaxSec=86400',
               '--property=TimeoutStopSec=15', '--property=KillMode=control-group',
               '--property=Restart=no', '--property=WorkingDirectory=' + str(root),
               '--property=UMask=0022', '--setenv=CUDA_VISIBLE_DEVICES=' + ','.join(EXPECTED.values()),
               '--setenv=CUDA_DEVICE_ORDER=PCI_BUS_ID', '--setenv=PYTHONUNBUFFERED=1',
               '--setenv=OMP_NUM_THREADS=4', '--setenv=MKL_NUM_THREADS=4',
               '--setenv=NCCL_P2P_DISABLE=1', '--setenv=TORCH_NCCL_ASYNC_ERROR_HANDLING=1',
               '/bin/bash', '-c', wrapped]
    subprocess.run(command, check=True)
    receipt = dict(unit=a.unit + '.service', gpu_indices=list(EXPECTED), gpu_uuids=list(EXPECTED.values()),
                   python=a.python, batch_size_per_node=2048, micro_batch_per_gpu=128, accumulation_steps=4,
                   duration_seconds=86400, train_sha256=a.sha256, output=a.output,
                   gate_sha256=hashlib.sha256(gate_text.encode()).hexdigest(),
                   launched_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (root / 'launch-hkust4.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
