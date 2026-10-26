"""随机输入、随机标签的 ResNet50 训练；每台机器有效 batch=1024。"""
import argparse
import contextlib
import datetime
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torchvision.models import resnet50


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--micro-batch', type=int, default=64)
    parser.add_argument('--seconds', type=int, default=86400)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world = int(os.environ['WORLD_SIZE'])
    assert args.batch_size % (world * args.micro_batch) == 0
    accumulation = args.batch_size // (world * args.micro_batch)
    torch.set_num_threads(4)
    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=5))
    torch.manual_seed(20261026 + rank)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device('cuda', local_rank)
    # weights=None：随机初始化，不下载任何预训练权重或数据集。
    model = resnet50(weights=None).to(device, memory_format=torch.channels_last)
    model = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    step = 0
    parameter = next(model.parameters())
    # 所有 rank 使用相同的结束判断，避免部分 rank 提前退出通信。
    while True:
        stop = torch.tensor(int(time.monotonic() - started >= args.seconds), device=device)
        dist.all_reduce(stop, op=dist.ReduceOp.MAX)
        if stop.item():
            break
        t0 = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        before = parameter.detach().clone()
        total_loss = torch.zeros((), device=device)
        for micro in range(accumulation):
            # 每个微批次独立生成噪声和标签，标签数与输入样本数一致。
            images = torch.randn(args.micro_batch, 3, 224, 224, device=device)
            images = images.contiguous(memory_format=torch.channels_last)
            labels = torch.randint(0, 1000, (args.micro_batch,), device=device)
            sync = contextlib.nullcontext() if micro == accumulation - 1 else model.no_sync()
            with sync:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    loss = torch.nn.functional.cross_entropy(model(images), labels)
                (loss / accumulation).backward()
            total_loss += loss.detach() / accumulation
        # DDP 平均各卡梯度，累积后每次更新恰好使用每台 1024 个样本。
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        if not torch.isfinite(grad_norm).item():
            raise RuntimeError('梯度出现非有限值')
        optimizer.step()
        torch.cuda.synchronize()
        step += 1
        dist.all_reduce(total_loss)
        total_loss /= world
        delta = (parameter.detach() - before).abs().max().item()
        if not torch.isfinite(total_loss).item() or (step <= 3 and delta == 0):
            raise RuntimeError('损失非有限值或参数没有实际更新')
        record = dict(
            utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            pid=os.getpid(), uid=os.getuid(), rank=rank, world_size=world,
            step=step, optimizer_steps=step, effective_batch_size=args.batch_size,
            micro_batch_per_gpu=args.micro_batch, accumulation_steps=accumulation,
            loss=total_loss.item(), parameter_max_abs_update=delta,
            step_seconds=round(time.monotonic() - t0, 4),
            elapsed_seconds=round(time.monotonic() - started, 3),
            allocated_MiB=round(torch.cuda.memory_allocated() / 2**20, 1),
            reserved_MiB=round(torch.cuda.memory_reserved() / 2**20, 1),
        )
        # 小型进度文件原子替换；训练日志按间隔输出，避免无限膨胀。
        if step <= 3 or step % 10 == 0:
            temp = output / ('rank-%d.json.tmp' % rank)
            temp.write_text(json.dumps(record) + '\n')
            temp.replace(output / ('rank-%d.json' % rank))
            if rank == 0:
                print(json.dumps(record), flush=True)
    dist.destroy_process_group()
    (output / ('rank-%d-complete.json' % rank)).write_text(json.dumps(record) + '\n')


if __name__ == '__main__':
    main()
