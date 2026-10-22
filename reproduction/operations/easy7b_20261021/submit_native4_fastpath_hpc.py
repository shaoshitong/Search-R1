"""Independent four-H100 fixed-work A/B and one-H100 compact kernel probe."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time


def main():
    request = json.loads(sys.stdin.read())
    base = Path(request['base'])
    source = Path(request['source_base'])
    commit = request['commit']
    sys.path.insert(0, str(base/'source'/commit))
    from reproduction.easy2560_joint4.control import own_base, publish
    from reproduction.easy2560_joint4.flash3_runtime import binding
    own_base(base)
    results = []
    for mode in request.get('modes', ['legacy', 'leancompact', 'compact-kernel']):
        if mode not in ('legacy', 'leancompact', 'compact-kernel'):
            raise ValueError('Unknown independent experiment mode')
        label = request.get('suffix', 'r1')
        root = base/'diagnostics'/('native4-'+mode+'-'+label)
        root.mkdir(parents=True, exist_ok=True)
        claim = root/'submit-claim.json'
        if claim.exists():
            results.append(dict(mode=mode, status='ALREADY_CLAIMED',
                receipt=json.loads((root/'submit-receipt.json').read_text())
                if (root/'submit-receipt.json').is_file() else None))
            continue
        common = ['env', 'LD_PRELOAD='+str(source/'preparation/flash3-runtime/libcudart.so.12'),
            'PYTHONDONTWRITEBYTECODE=1', 'PYTHONUNBUFFERED=1', 'OMP_NUM_THREADS=8',
            'HF_HUB_OFFLINE=1', 'TRANSFORMERS_OFFLINE=1', 'NCCL_NVLS_ENABLE=0',
            'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True',
            'TRITON_CACHE_DIR='+str(root/'cache/triton'), 'XDG_CACHE_HOME='+str(root/'cache'),
            str(source/'env/bin/python'), '-B']
        if mode == 'compact-kernel':
            publish(root/'logs/run-manifest.json', dict(training_world_size=4,
                source_commit=commit, infra_flash_attention_3=binding(source)))
            code = base/'source'/commit/'reproduction/operations/easy7b_20261021/qualify_fa3_compact_gqa_cuda.py'
            # -m receives this source first in its subprocess environment.
            command = common + ['-c', 'import runpy,sys;sys.path.insert(0,'+
                repr(str(base/'source'/commit))+');sys.argv='+repr([str(code),'--base',str(root)])+
                ';runpy.run_path('+repr(str(code))+',run_name="__main__")']
            gpus, cpus, memory, duration = 1, 8, '64G', '00:30:00'
        else:
            code = base/'source'/commit/'reproduction/operations/easy7b_20261021/benchmark_native4_fastpath.py'
            command = common + ['-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
                '--nproc_per_node=4', str(code), '--source-base', str(source),
                '--checkpoint', request['checkpoint'], '--mode', mode,
                '--output', str(root), '--steps', '20']
            gpus, cpus, memory, duration = 4, 48, '512G', '02:00:00'
        argv = ['sbatch', '--parsable', '--partition=emergency_acd', '--qos=formal-user',
            '--account=jxu957', '--nodes=1', '--ntasks=1', '--gres=gpu:'+str(gpus),
            '--cpus-per-task='+str(cpus), '--mem='+memory, '--time='+duration,
            '--job-name=easy7b-'+mode, '--export=NONE',
            '--chdir='+str(base/'source'/commit),
            '--output='+str(root/'slurm-%j.log'), '--error='+str(root/'slurm-%j.log'),
            '--wrap='+shlex.join(command)]
        publish(claim, dict(source_commit=commit, command=argv, mode=mode,
            checkpoint=request['checkpoint'], production_modified=False, created_unix=time.time()))
        result = subprocess.run(argv, capture_output=True, text=True, timeout=30)
        receipt = dict(source_commit=commit, mode=mode, root=str(root),
            returncode=result.returncode, stdout=result.stdout, stderr=result.stderr,
            checked_unix=time.time(), production_modified=False)
        if result.returncode == 0:
            job = result.stdout.strip().split(';')[0]
            if not job.isdecimal():
                raise RuntimeError('Unknown sbatch outcome; inspect existing claim')
            receipt['job'] = job
            with (root/'job.id').open('x') as f:
                f.write(job+'\n');f.flush();os.fsync(f.fileno())
        publish(root/'submit-receipt.json', receipt)
        results.append(receipt)
    print(json.dumps(results), flush=True)


if __name__ == '__main__':
    main()
