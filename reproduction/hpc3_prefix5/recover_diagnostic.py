"""Resume the genuine P10 diagnostic fault from the qualified complete CP9.

Runs only inside one new four-GPU Slurm allocation. The failed source remains
immutable; CPU qualification and explicit joint import precede all CUDA work.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import uuid

from .runtime import shared_devices, visible_allocation, sha, once


def prepare(base):
    """Stage a terminal-only CP9 import; never submit or start a GPU process."""
    from .terminal import writers_present
    from reproduction.prefix_selection200.i1_i6_joint_resume import (
        source_manifest, compatibility_proof)
    base = Path(base).resolve(strict=True)
    source = Path('/data/user/jxu957/experiments/dynrpo-prefix5-thread5-20261017')
    if base != Path(str(source) + '-r2'):
        raise ValueError('Require the isolated CP9 continuation namespace')
    p = base / 'preparation'
    template = p / 'resume-config-template.json'
    if template.exists():
        return dict(status='EXISTING_PREPARED_CP9_CONTINUATION', path=str(template))
    c = json.loads((source / 'preparation/launch-config.json').read_text())
    joint = json.loads((source / 'latest-joint.json').read_text())
    ledger = json.loads((source / 'logs/process-ledger.json').read_text())
    actor = json.loads((source / 'actor/latest.json').read_text())
    probe = json.loads((source / 'probe/latest.json').read_text())
    if (joint['outer'] != 9 or actor != joint['actor'] or probe != joint['probe']
            or actor['actual_adam_steps_per_rank'] != [180] * 4
            or probe['actual_adam_steps_per_rank'] != [0] * 4
            or any(row['state'] != 'STOPPED' for row in ledger['jobs'].values())
            or list((source / 'actor/outers').glob('outer-0010-attempt-*'))
            or (source / 'outers/outer-000010/probe-update').exists()):
        raise ValueError('Only the genuine diagnostic fault before any outer10 optimizer write is admitted')
    pending = {name for name in ledger['jobs'] if name.startswith('outer-000010-')}
    if not pending <= {'outer-000010-prefix', 'outer-000010-score',
                       'outer-000010-complete-unselected-pool',
                       'outer-000010-current-actor-diagnostic'}:
        raise ValueError('An unaccounted outer10 phase exists; do not discard optimizer work')
    births = {}
    for job in ledger['jobs'].values():
        for row in job.get('ledger', {}).values():
            key = tuple(row[k] for k in ('pid', 'start_ticks', 'sid', 'uid'))
            births[key] = {k: row[k] for k in ('pid', 'start_ticks', 'sid', 'uid')}
    terminal = dict(run_id=c['run_id'], execution_id=c['execution_id'],
        source_commit=c['source_commit'], base=str(source), execution_backend='slurm_hpc3',
        slurm_job=str(c['slurm_job']), uid=os.getuid(), host=c['slurm_node'],
        termination_cause='FAULT_EXIT', pipeline_alive=False, owned_writers_gone=True,
        actual_births_present=[], owned_births=list(births.values()),
        absence_authority='fresh terminal Slurm allocation, batch and extern/cgroup accounting',
        native_host_boot_not_sampled=True,
        process_ledger_sha256=sha(source / 'logs/process-ledger.json'),
        source_config_sha256=sha(source / 'preparation/launch-config.json'),
        uncommitted_after_joint=dict(actor_actual_writes_per_rank=[0] * 4,
            probe_actual_writes_per_rank=[0] * 4, optimizer_work_in_flight=False,
            all_updates_accounted=True), observed_unix=time.time())
    writers_present(terminal, c)
    terminal_path = source / 'preparation/slurm-fault-729343-cp9.json'
    if terminal_path.exists():
        terminal = json.loads(terminal_path.read_text())
        writers_present(terminal, c)
        if terminal['process_ledger_sha256'] != sha(source / 'logs/process-ledger.json'):
            raise ValueError('Failed source ledger changed since its terminal receipt')
    else:
        once(terminal_path, terminal)
    publication = json.loads((p / 'source-publication.json').read_text())
    parent = source_manifest(source / 'repo', c['source_commit'])
    target = source_manifest(base / 'repo', publication['commit'],
                             science_source_commit=c['source_commit'])
    compatibility = compatibility_proof(parent, target)
    compat_path = p / 'source-compatibility.json'
    once(compat_path, compatibility)
    spec = dict(adapter='i1_i6_joint_v1', source_base=str(source),
        source_commit=c['source_commit'], source_execution_id=c['execution_id'],
        source_config_sha256=sha(source / 'preparation/launch-config.json'),
        source_joint_sha256=sha(source / 'latest-joint.json'), completed_outer=9,
        terminal_proof=str(terminal_path), terminal_proof_sha256=sha(terminal_path),
        qualification_proof=str(source / 'preparation/cp9-actual-full-state-qualification.json'))
    c.update(base=str(base), execution_id=uuid.uuid4().hex, source_commit=publication['commit'],
        science_source_commit=parent['science_source_commit'], source_compatibility=str(compat_path),
        source_compatibility_sha256=sha(compat_path), resume_joint=spec,
        environment_root=str(base / 'env'), retriever_gpu_lib=str(base / 'retriever-gpu-lib'),
        pool_state_path=str(base / 'question-pool/state.json'),
        prefix_probe_receipt=str(base / 'probe/latest.json'))
    once(template, c)
    return dict(status='TERMINAL_CP9_CONTINUATION_PREPARED_NO_GPU_STARTED',
                completed_outer=9, actor_adam=[180] * 4, probe_adam=[0] * 4,
                source_commit=publication['commit'], science_source_commit=parent['science_source_commit'],
                files_changed=sorted(compatibility['controller_changes']))


def run(base):
    from reproduction.prefix_selection200.i1_i6_joint_resume import import_complete
    from reproduction.prefix_selection200.launch import run as launch, validate_config
    import csv
    import io
    base = Path(base).resolve(strict=True)
    expected = Path('/data/user/jxu957/experiments/dynrpo-prefix5-thread5-20261017-r2')
    if base != expected or not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Require the isolated CP9 continuation and its actual Slurm allocation')
    p = base / 'preparation'
    if (p / 'submit-intent.json').exists() or (base / 'logs/process-ledger.json').exists():
        raise ValueError('An existing continuation must be inspected, never blindly replayed')
    c = json.loads((p / 'resume-config-template.json').read_text())
    spec = c['resume_joint']
    if spec['completed_outer'] != 9 or spec['source_base'] != str(expected).removesuffix('-r2'):
        raise ValueError('Only the actual complete CP9 boundary is authorized by this entry')
    terminal = json.loads(Path(spec['terminal_proof']).read_text())
    original = json.loads((Path(spec['source_base']) / 'preparation/launch-config.json').read_text())
    from .terminal import writers_present
    writers_present(terminal, original)
    publication = json.loads((p / 'source-publication.json').read_text())
    if publication['commit'] != c['source_commit'] or publication['remote_sha'] != c['source_commit']:
        raise ValueError('Continuation source is not its pushed publication')
    for name, digest in publication['files'].items():
        if sha(base / 'repo' / name) != digest:
            raise ValueError('Published continuation source SHA differs: ' + name)
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    assigned = os.environ.get('SLURM_JOB_GPUS', '').split(',')
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid',
                                   '--format=csv,noheader,nounits'], text=True)
    by_index = {int(row[0]): row[1].strip() for row in csv.reader(io.StringIO(raw))}
    selected = visible_allocation(visible, assigned, by_index)
    c.update(shared_devices(selected, by_index), slurm_job=os.environ['SLURM_JOB_ID'],
             slurm_node=os.uname().nodename)
    python = str(base / 'env/bin/python')
    proof = Path(spec['qualification_proof'])
    cpu_env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2',
                   MKL_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1')
    subprocess.run([python, '-m', 'reproduction.prefix_selection200.i1_i6_checkpoint_cpu',
        '--base', spec['source_base'], '--outer', '9',
        '--terminal-proof', spec['terminal_proof'],
        '--terminal-proof-sha256', spec['terminal_proof_sha256'],
        '--output', str(proof)], env=cpu_env, check=True)
    qualified = json.loads(proof.read_text())
    if (qualified['completed_outer'] != 9 or qualified['actual_adam_steps_per_rank'] != [180] * 4
            or qualified['probe_adam_steps_per_rank'] != [0] * 4):
        raise ValueError('Actual qualified full state differs from CP9')
    spec['qualification_proof_sha256'] = sha(proof)
    validate_config(c, base)
    once(p / 'launch-config.json', c)
    once(p / 'submit-intent.json', dict(config_sha256=sha(p / 'launch-config.json'),
        run_id=c['run_id'], source_commit=c['source_commit'], job=os.environ['SLURM_JOB_ID']))
    once(base / 'logs/slurm-allocation.json', dict(job=os.environ['SLURM_JOB_ID'],
        node=os.uname().nodename, uid=os.getuid(), training=selected, auxiliary=selected,
        assigned_uuids=[by_index[i] for i in selected], total_physical_gpus=4,
        gpu_resource_layout=c['gpu_resource_layout']))
    print(json.dumps(import_complete(base)), flush=True)
    return launch(base)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    print(json.dumps(prepare(args.base) if args.prepare else run(args.base)), flush=True)
