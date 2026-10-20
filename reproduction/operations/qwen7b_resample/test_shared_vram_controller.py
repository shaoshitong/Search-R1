"""CPU-only TP2/TP4 controller simulations; no CUDA, remote calls, or real children.

Run from the repository root with::

    python3 -m unittest reproduction.operations.qwen7b_resample.test_shared_vram_controller
"""

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import time
import types
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).with_name('shared_vram_worker.py').resolve()
SPEC = importlib.util.spec_from_file_location('shared_vram_controller_under_test', SOURCE)
WORKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WORKER)


class SharedVramControllerTests(unittest.TestCase):
    def simulate(self, mode, model, retrieval, tp=2):
        with tempfile.TemporaryDirectory(prefix='q7b-two-engine-cpu-') as temporary:
            base = Path(temporary)
            evidence = base / 'evidence.json'
            evidence.write_text(json.dumps(dict(captured_epoch=time.time(),
                resource_limits=dict(runtime_max_seconds=28800))))

            def read(path):
                return json.loads(Path(path).read_text())

            def publish(path, value):
                path = Path(path)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value))

            pool = types.SimpleNamespace(
                configuration=lambda _: dict(base=str(base), retrieval_index='index',
                    retrieval_corpus='corpus', retrieval_data_cache='data'),
                handoff=lambda *_: dict(test=True), publish=publish, read=read,
                sha=lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                RUN_ID='70188b8e50fd4b45bc420ce58014f46e',
                URL='http://127.0.0.1:28831/retrieve')
            binding = dict(job_id='shared5-101', model_indices=model,
                model_cuda_uuids=['GPU-' + str(index) for index in model],
                model_cuda_visible_devices=','.join(map(str, model)),
                retrieval_indices=retrieval,
                retrieval_cuda_uuids=['GPU-' + str(index) for index in retrieval],
                retrieval_cuda_visible_devices=','.join(map(str, retrieval)),
                actual_cuda_runtime_uuids=['GPU-' + str(index) for index in model + retrieval],
                native_unit='test-own.service', original_cuda_visible_devices='3,4',
                tensor_parallel_size=tp, model_engine_count=len(model) // tp)
            args = argparse.Namespace(base=str(base), handoff_marker=None,
                allocation_evidence=str(evidence), pool_worker=str(SOURCE.with_name('pool_worker.py')),
                retriever_script=str(SOURCE.with_name('shared_vram_retriever.py')),
                run_seconds=60, batch_reserve_seconds=1, encoder=None,
                retrieval_weights='15,9,20,20', retrieval_shards=64,
                model_max_seqs=128, model_batch_tokens=16384, model_kv_block_cap=16384, tp=tp)
            processes, launches, stopped = [], [], []
            test = self

            class FakeProcess:
                def __init__(self, argv, env, **_):
                    self.pid = 1000 + len(processes)
                    self.returncode = None
                    self.polls = 0
                    self.sampler = '--sampler-loop' in argv
                    processes.append(self)
                    if not self.sampler:
                        return

                    def flag(name):
                        return argv[argv.index(name) + 1]

                    self.directory = Path(flag('--job-dir'))
                    self.launch = Path(flag('--launch-dir'))
                    self.ordinal = int(env['Q7B_SHARED_RETRY_ORDINAL'])
                    self.binding = read(self.directory / 'allocation.json')
                    self.engine = self.binding.get('engine_index', 0)
                    test.assertEqual(env['Q7B_POOL_INTERNAL'], str(self.directory))
                    test.assertEqual(env['CUDA_VISIBLE_DEVICES'],
                        ','.join(map(str, model[tp * self.engine:tp * self.engine + tp])))
                    test.assertEqual(flag('--tensor-parallel-size'), str(tp))
                    test.assertEqual(flag('--model-max-seqs'), '128')
                    test.assertEqual(flag('--model-batch-tokens'), '16384')
                    test.assertEqual(flag('--model-kv-block-cap'), '16384')
                    launches.append((self.engine, self.ordinal))

                def poll(self):
                    if self.returncode is not None or not self.sampler:
                        return self.returncode
                    if len(model) // tp > 1:
                        test.assertGreaterEqual(sum(process.sampler for process in processes), 2,
                            'Both engines must start before either sampler is polled')
                    self.polls += 1
                    if self.polls < 2:
                        return None
                    if self.engine == 0 and self.ordinal == 0 and mode in ('local-oom', 'failure'):
                        if mode == 'local-oom':
                            publish(self.launch / 'typed-model-initialization-oom.json',
                                dict(actual_typed_cuda_oom=True, before_first_new_sampling_request=True))
                        self.returncode = 1
                        return self.returncode
                    result = dict(status='POOL_INPUT_SCAN_FINISHED', run_id=pool.RUN_ID,
                        job_id=self.binding['job_id'], completed_own_batches=1,
                        questions=128, trajectories=640)
                    publish(self.directory / 'progress.json', result)
                    publish(self.directory / 'sampler-exit.json', result)
                    self.returncode = 0
                    return self.returncode

            def stop(process):
                stopped.append(process.pid)
                if process.returncode is None:
                    process.returncode = -15

            pool.stop_group = stop

            def ready(*_, **__):
                return io.BytesIO(json.dumps(dict(ready=True, pid=processes[0].pid,
                    run_id=pool.RUN_ID, gpu_shards=len(retrieval), logical_flat_shards=64,
                    encoder_dtype='float16', index_dtype='float16',
                    actual_cuda_uuids=binding['retrieval_cuda_uuids'],
                    original_global_ids=True, endpoint=pool.URL)).encode())

            failure = None
            with patch.object(WORKER, 'admit', return_value=(binding, [])), \
                    patch.object(WORKER.subprocess, 'Popen', side_effect=FakeProcess), \
                    patch('urllib.request.urlopen', side_effect=ready), \
                    patch.object(WORKER.time, 'sleep', return_value=None), \
                    contextlib.redirect_stdout(io.StringIO()):
                try:
                    WORKER.controller(pool, args)
                except ValueError as error:
                    failure = str(error)

            parent = next((base / 'pool/jobs/job-shared5-101').glob('attempt-*'))
            self.assertTrue(all(process.pid in stopped for process in processes),
                'Every created own process must be reaped')
            if mode == 'failure':
                self.assertIsNotNone(failure)
                self.assertIn('no eligible local typed constructor OOM retry', failure)
                self.assertTrue((parent / 'failure.json').exists())
                self.assertFalse((parent / 'complete.json').exists())
                return

            self.assertIsNone(failure)
            result = read(parent / 'complete.json')
            self.assertEqual(result['completed_own_batches'], len(model) // tp)
            self.assertEqual(result['trajectories'], 640 * (len(model) // tp))
            self.assertEqual(result['sampler_returncodes'], [0] * (len(model) // tp))
            self.assertEqual(result['tensor_parallel_size'], tp)
            if mode == 'local-oom':
                self.assertEqual(launches, [(0, 0), (1, 0), (0, 1)])
            if len(model) // tp > 1:
                directories = [next((base / ('pool/jobs/job-shared5-101-engine-%02d' % index)).glob('attempt-*'))
                    for index in range(2)]
                self.assertTrue(all((directory / 'complete.json').exists() for directory in directories))
                self.assertEqual(read(directories[0] / 'allocation.json')['model_indices'], [3, 4])
                self.assertEqual(read(directories[1] / 'allocation.json')['model_indices'], [6, 7])
            else:
                self.assertEqual(result['sampler_returncode'], 0)
                self.assertEqual(launches, [(0, 0)])
                self.assertEqual(read(parent / 'allocation.json')['model_indices'], model)
                self.assertEqual(read(parent / 'engines.json')['engine_count'], 1)

    def test_two_engines_progress_in_parallel(self):
        self.simulate('success', [3, 4, 6, 7], [0, 1, 2, 5])

    def test_constructor_oom_retries_only_its_own_engine(self):
        self.simulate('local-oom', [3, 4, 6, 7], [0, 1, 2, 5])

    def test_single_engine_compatibility(self):
        self.simulate('single', [3, 4], [0, 2, 5, 6, 7])

    def test_non_oom_failure_reaps_all_own_children(self):
        self.simulate('failure', [3, 4, 6, 7], [0, 1, 2, 5])

    def test_single_tp4_uses_all_four_model_gpus(self):
        self.simulate('single-tp4', [3, 4, 6, 7], [0, 1, 2, 5], tp=4)


if __name__ == '__main__':
    unittest.main()
