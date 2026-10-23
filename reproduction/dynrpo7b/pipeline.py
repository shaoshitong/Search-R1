"""DynRPO lifecycle using the proven native4 training/retrieval components."""
import fcntl
import json
import signal
import socket
from pathlib import Path

from reproduction.common8.run_pipeline import Pipeline as CommonPipeline
from reproduction.common8.run_pipeline import (
    supported_execution_role, assert_initial_state, request_interrupt,
)
from reproduction.common8.verify_data import digest
from .infra_profile import EXPORT_PORT, native4


class Pipeline(CommonPipeline):
    def __init__(self, base, run):
        super().__init__(base, run)
        from search_g1.service_ports import probe_url, retriever_url
        self.env.update(PROBE_URL=probe_url(run=run), RETRIEVER_URL=retriever_url(run))

    def sample(self):
        """No continuous nvidia-smi subprocesses in the training hot path.

        Actual worker timings, CUDA memory and optimizer writes already appear
        in the ordinary training metrics; scheduler/Slurm observation is external.
        """
        return

    def train_and_verify(self):
        self.stage('DYNRPO7B_GRPO_1_TO_200')
        log = Path(self.run['train_log'])
        if log.exists():
            raise RuntimeError('Canonical trainer log belongs to an existing attempt')
        process = self.start('trainer', ['bash', self.train_script], self.training_gpus,
                             {'SEARCH_G1_TRAIN_LOG': str(log)})
        self.wait('trainer', process)
        self.stage('DYNRPO7B_FINAL_CPU_ACCEPTANCE')
        acceptance = self.start('final-acceptance', [self.python, '-m',
            'reproduction.dynrpo7b.final_acceptance', '--base', str(self.base)], [])
        self.wait('final-acceptance', acceptance)
        return 'FINISHED_DYNRPO7B_200_FINAL51713_AND_OWNED_CLEANUP'


def launch(base, run):
    base = Path(base)
    if not native4(run):
        raise ValueError('Pipeline requires explicit independent DynRPO native4')
    supported_execution_role(run, base)
    for name in ('logs', 'tmp', 'ray', 'cache/triton', 'cache/inductor', 'cache/hf', 'artifacts'):
        (base / name).mkdir(parents=True, exist_ok=True)
    lock = (base / 'gpu-job.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    signal.signal(signal.SIGTERM, request_interrupt)
    signal.signal(signal.SIGINT, request_interrupt)
    assert_initial_state(run)
    if (base / 'logs/process-ledger.json').exists():
        raise RuntimeError('An existing pipeline attempt must be resumed explicitly')
    for name, expected in run['source_sha256'].items():
        if digest(base / 'repo' / name) != expected:
            raise ValueError('Published runtime source changed: ' + name)
    from search_g1.service_ports import ports
    for port in ports(run).values():
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', port))
    # Resource UUID mapping once, no separate forward/backward GPU qualification.
    from reproduction.common8.native_allocation import admit
    admit(base)
    # Header-only disk allowance, instead of rereading all official 7B shards
    # which fresh preparation has already SHA-verified.
    import shutil
    from reproduction.common8.disk_admission import checkpoint_floor
    from .deploy import publish
    allowance = checkpoint_floor(Path(run['initial_model']))
    allowance.update(run_id=run['run_id'], actual_free_bytes=shutil.disk_usage(base).free)
    allowance['accepted'] = allowance['actual_free_bytes'] >= allowance['required_free_bytes']
    publish(base / 'logs/checkpoint-disk-admission.json', allowance)
    if not allowance['accepted']:
        raise RuntimeError('Shared storage lacks the required checkpoint allowance')
    Pipeline(base, run).execute(EXPORT_PORT)
