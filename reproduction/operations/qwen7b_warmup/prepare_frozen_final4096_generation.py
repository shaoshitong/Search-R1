"""Reuse the frozen, ungenerated final roster and mature two-GPU collector.

Run prepare/status on HPC3 through the existing private control. This controller
never submits, generates, reads final labels, or scores a probe. Root submits
the four resulting array entries. Each array is 0-3%1: at most eight GPUs total
when all four are submitted. Existing precommitted questions/run IDs/G5 stay.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

B = Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019')
A = B/'algorithm-improvement-20261021'
MATERIALIZATION = B/'confirmation-materialization-deploy-20261021/materialized/materialization.json'
MATERIALIZATION_SHA = 'ff68f243b1daa7d8a617a2c7a825e1126d0771eee2e92b96a85dc5f9b4f2bfe3'
FREEZE_SHA = 'f1f00560ca8950be5b157c409d3932744be99e36952a76cbdf0cf75dc95001d3'
BASE = A/'frozen-final4096-generation-20261022-v1'
REVISION = 'a09a35458c702b33eeacc393d103063234e8bc28'


def read(path):
    return json.loads(Path(path).read_bytes())


def proof(path):
    path = Path(path)
    raw = path.read_bytes()
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def checked(ref):
    path = Path(ref['path'])
    if proof(path) != {key:ref[key] for key in ('path','bytes','sha256')}:
        raise ValueError('Changed prepared metadata/source: '+str(path))
    return path


def once(path, value):
    path = Path(path)
    raw = (json.dumps(value, sort_keys=True, separators=(',', ':'))+'\n').encode()
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError('Preserve conflicting prior preparation: '+str(path))
    else:
        with path.open('xb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())


def render(config, output, code, job):
    # Actor and retriever are separate devices inside this exact old collector.
    # Full journals preserve later P40/P80 causal extraction possibilities.
    runtime = code/'reproduction/operations/qwen7b_warmup/runtime.sh'
    return '''#!/usr/bin/env bash
#SBATCH --job-name=@@JOB@@
#SBATCH --partition=acd_u
#SBATCH --qos=normal
#SBATCH --account=mou176
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH --array=0-3%1
#SBATCH --output=@@OUTPUT@@/sampling-%A_%a.log
set -euo pipefail
export QWEN7B_CODE_ROOT=@@CODE@@
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
# Fresh actor SHA and immutable preparation check run inside this allocation.
bash @@RUNTIME@@ -u -m reproduction.operations.qwen7b_warmup.confirmation_materialize --check-event-config @@CONFIG@@
srun --kill-on-bad-exit=1 bash @@RUNTIME@@ -u -m reproduction.operations.qwen7b_warmup.shard --config @@CONFIG@@ --shard "$SLURM_ARRAY_TASK_ID"
'''.replace('@@JOB@@', job).replace('@@OUTPUT@@', str(output)).replace(
        '@@CODE@@', shlex.quote(str(code))).replace('@@RUNTIME@@', shlex.quote(str(runtime))).replace(
        '@@CONFIG@@', shlex.quote(str(config)))


def prepare():
    BASE.mkdir(exist_ok=True)
    guard = (BASE/'prepare.lock').open('a+')
    fcntl.flock(guard, fcntl.LOCK_EX|fcntl.LOCK_NB)
    done = BASE/'prepared.json'
    if done.exists():
        prior = read(done)
        for entry in prior['arrays']:
            checked(entry['config'])
            checked(entry['entry'])
        return prior
    if proof(MATERIALIZATION)['sha256'] != MATERIALIZATION_SHA:
        raise ValueError('Wrong original frozen final materialization')
    materialized = read(MATERIALIZATION)
    if (materialized['status'] != 'CONFIRMATION_STABILITY_TWO_REAL_G5_EVENT_INPUTS_PREPARED_NOT_GENERATED'
            or materialized['source_freeze']['sha256'] != FREEZE_SHA
            or materialized['fit_or_tuning_allowed'] is not False):
        raise ValueError('Only the original untouched fixed final roster is allowed')
    checked(materialized['source_freeze'])
    publication_path = MATERIALIZATION.parents[1]/'source-publication.json'
    publication = read(publication_path)
    code = Path(publication['code_root'])
    arrays = []
    for pool_name in ('confirmation2048', 'stability2048'):
        pool = materialized['pools'][pool_name]
        if pool['questions'] != 2048:
            raise ValueError('The original two disjoint2048 pools must remain intact')
        checked(pool['planned_G10_and_fixed_G5'])
        checked(pool['question_plan'])
        for event_index, ref in enumerate(pool['event_configs']):
            c = read(checked(ref))
            if (c['model_revision'] != REVISION or c['code_root'] != str(code)
                    or c['evaluation_only'] is not True or c['fitting_authorized'] is not False
                    or c['fit_epochs'] != 0 or c['actor_optimizer_updates'] != 0
                    or c['gpus_per_sampling_job'] != 2 or c['sampling_shards'] != 4
                    or c['capture_prefix_tokens'] != 20 or c['question_max_tokens'] != 273):
                raise ValueError('Frozen official actor/collector contract differs')
            for name, sha in c['collector_code_files_sha256'].items():
                if proof(code/name)['sha256'] != sha:
                    raise ValueError('Published old collector source changed: '+name)
            for source in c['immutable_preparation_sources']:
                checked(source)
            # Metadata only: never read or hash model/index/tensor payload here.
            checked(c['actor_complete'])
            letter = 'AB'[event_index]
            key = pool_name+'-'+letter
            output = BASE/key
            output.mkdir(exist_ok=True)
            job = 'q7b-final-'+('c' if pool_name == 'confirmation2048' else 's')+letter
            text = render(Path(ref['path']), output, code, job)
            entry = output/'sampling.sbatch'
            if entry.exists() and entry.read_text() != text:
                raise ValueError('Different existing final generation entry')
            if not entry.exists():
                entry.write_text(text)
            subprocess.run(['bash', '-n', str(entry)], check=True)
            arrays.append(dict(key=key, job_name=job, entry=proof(entry), config=ref,
                original_event_root=c['base'], run_id=c['run_id'],
                sampling_attempt_id=c['sampling_attempt_id'], array='0-3%1',
                resources=dict(GPUs_per_task=2, max_simultaneous_tasks=1,
                    actor_GPUs=1, retriever_GPUs=1, CPUs=16, memory_GiB=128),
                questions_per_task=512, G5_trajectories_per_task=2560,
                submission_command=['sbatch', '--parsable', str(entry)]))
    # Test only the old already-published critical import closure, with no GPU.
    environment = dict(os.environ, QWEN7B_CODE_ROOT=str(code), CUDA_VISIBLE_DEVICES='',
        PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    script = ('import reproduction.operations.qwen7b_warmup.shard; '
        'import reproduction.operations.qwen7b_warmup.confirmation_materialize; '
        'print("FROZEN_FINAL_COLLECTOR_CPU_IMPORT_PASS")')
    imported = subprocess.run(['bash', str(code/'reproduction/operations/qwen7b_warmup/runtime.sh'),
        '-c', script], env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, timeout=40)
    if imported.returncode or 'FROZEN_FINAL_COLLECTOR_CPU_IMPORT_PASS' not in imported.stdout:
        raise RuntimeError('Published mature collector import failed: '+imported.stderr[-1500:])
    result = dict(status='FOUR_FROZEN_FINAL_G5_GENERATION_ARRAYS_PREPARED_NOT_SUBMITTED',
        materialization=proof(MATERIALIZATION), source_publication=proof(publication_path),
        collector_commit=publication['commit'], code_root=str(code),
        original_frozen_pool_manifest=materialized['source_freeze'],
        arrays=arrays, questions=4096, real_G10_trajectories_to_generate=40960,
        fixed_G5_confirmation_trajectories=20480,
        max_simultaneous_two_GPU_tasks=4, max_simultaneous_GPUs=8,
        normal_acd_u=True, emergency_used=False, no_dependency=True,
        original_plans_run_ids_and_G5_preserved=True,
        mature_collector_CPU_import_passed=True, bash_syntax_passed=True,
        model_or_checkpoint_or_feature_payload_read=False,
        generated_final_labels_or_predictions_read=False,
        actor_training=False, probe_training_or_scoring=False,
        root_must_check_same_name_jobs_before_single_submission=True,
        submitted=False, prepared_epoch=time.time())
    once(done, result)
    return result


def status():
    prepared = read(BASE/'prepared.json')
    result = dict(status='FROZEN_FINAL_GENERATION_IDENTITY_AND_BATCH_COUNTS_ONLY', arrays=[])
    for entry in prepared['arrays']:
        shards = []
        for index in range(4):
            base = Path(entry['original_event_root'])/'sampling'/('shard-%02d' % index)
            row = dict(shard=index, complete_exists=(base/'complete.json').exists())
            # Counters only, never aggregate ownEM or any prediction.
            if (base/'progress.json').exists():
                p = read(base/'progress.json')
                row['progress'] = {key:p[key] for key in ('questions','trajectories','completed_batches','total_batches','epoch') if key in p}
            if (base/'controller.json').exists():
                p = read(base/'controller.json')
                row['controller'] = {key:p[key] for key in ('pid','start_ticks','job_id','epoch') if key in p}
            row['failure_exists'] = (base/'failure.json').exists()
            shards.append(row)
        result['arrays'].append(dict(key=entry['key'], job_name=entry['job_name'], shards=shards))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', choices=('prepare', 'status'), default='prepare')
    args = parser.parse_args()
    print(json.dumps(prepare() if args.action == 'prepare' else status()))


if __name__ == '__main__':
    main()
