"""CPU functional coverage for the consumer's complete controller lifecycle.

Uses a real temporary file tree and fake process/health/admission boundaries.
No GPU libraries, real child processes, HTTP connections or remote calls run.
"""

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch


DIRECTORY = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('relay_consumer_controller_under_test',
                                            DIRECTORY / 'relay_consumer_worker.py')
CONSUMER = importlib.util.module_from_spec(SPEC)
with patch.object(sys, 'path', [str(DIRECTORY)] + sys.path):
    SPEC.loader.exec_module(CONSUMER)


class RelayConsumerControllerTests(unittest.TestCase):
    def test_normal_lifecycle_seals_batch_and_reaps_only_local_children(self):
        self.run_lifecycle(reuse=False)

    def test_borrowed_relay_is_not_started_or_stopped(self):
        self.run_lifecycle(reuse=True)

    def test_tp1_is_forwarded_to_the_sampler(self):
        self.run_lifecycle(reuse=True, tp=1)

    def run_lifecycle(self, reuse, tp=4):
        with tempfile.TemporaryDirectory(prefix='q7b-consumer-controller-cpu-') as temporary:
            base = Path(temporary)

            def read(path):
                return json.loads(Path(path).read_text())

            def publish(path, value):
                path = Path(path)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value))

            run_id = '70188b8e50fd4b45bc420ce58014f46e'
            publish(base / 'config.json', dict(run_id=run_id, seed=20261021))
            publish(base / 'preparation/pool-handoff.json', dict(run_id=run_id, test_fixture=True))
            upstream = base / 'upstream-ready.json'
            upstream_pid = 1490465
            publish(upstream, dict(role='upstream', source_pid=upstream_pid,
                listen_url='http://192.0.2.5:32123', process=dict(pid=7000)))
            evidence = base / 'allocation-evidence.json'
            publish(evidence, dict(captured_epoch=time.time(),
                resource_limits=dict(runtime_max_seconds=28800)))
            token = base / 'token'
            token.write_text('private-test-fixture-token-never-used-for-real-http')
            token.chmod(0o600)
            pool = types.SimpleNamespace(
                configuration=lambda path: read(Path(path) / 'config.json'),
                handoff=lambda _, marker: read(marker), read=read, publish=publish,
                sha=lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                RUN_ID=run_id, URL='http://127.0.0.1:28831/retrieve')
            model = [2] if tp == 1 else [0, 1, 2, 3]
            numeric = ','.join(map(str, model))
            binding = dict(job_id='relay10-test', model_indices=model,
                model_cuda_uuids=['GPU-test-%d' % index for index in model], tensor_parallel_size=tp,
                model_cuda_visible_devices=numeric, shared_retriever_owned_by_consumer=False)
            args = argparse.Namespace(base=str(base), handoff_marker=None,
                pool_worker=str(DIRECTORY / 'pool_worker.py'),
                relay_script=str(DIRECTORY / 'retrieval_relay.py'), upstream_ready=str(upstream),
                token_file=str(token), allocation_evidence=str(evidence),
                run_seconds=60, batch_reserve_seconds=1, model_max_seqs=8, tp=tp,
                model_batch_tokens=4096, model_kv_block_cap=1024, model_reserve_gib=.25)
            if reuse:
                args.reuse_local_relay_ready = str(base / 'borrowed-ready.json')
                publish(args.reuse_local_relay_ready, dict(test_fixture=True))
            processes, stopped = [], []
            test = self

            class FakeProcess:
                def __init__(self, command, env, **options):
                    self.pid = 8001 + len(processes)
                    self.returncode = None
                    self.polls = 0
                    self.sampler = '--sampler-loop' in command
                    self.kind = 'sampler' if self.sampler else 'local-relay'
                    processes.append(self)
                    test.assertTrue(options['start_new_session'])

                    def flag(name):
                        return command[command.index(name) + 1]

                    if self.sampler:
                        self.jobdir = Path(flag('--job-dir'))
                        self.launch = Path(flag('--launch-dir'))
                        test.assertEqual(env['Q7B_POOL_INTERNAL'], str(self.jobdir))
                        test.assertEqual(env['CUDA_VISIBLE_DEVICES'], numeric)
                        test.assertEqual(flag('--tp'), str(tp))
                        # A functioning sampler writes into the supplied launch
                        # directory; it does not repair controller-owned parents.
                        (self.launch / 'model-initialized.json').write_text(json.dumps(
                            dict(test_fixture=True, tensor_parallel_size=tp)))
                    else:
                        test.assertIn('downstream', command)
                        test.assertEqual(flag('--upstream-url'), read(upstream)['listen_url'])
                        test.assertEqual(env['CUDA_VISIBLE_DEVICES'], '')
                        Path(flag('--ready-file')).write_text(json.dumps(dict(role='downstream', pid=self.pid)))

                def poll(self):
                    if not self.sampler or self.returncode is not None:
                        return self.returncode
                    self.polls += 1
                    if self.polls == 1:
                        return None
                    result = dict(status='POOL_BATCH_BOUNDARY_TIME_LIMIT', run_id=run_id,
                        job_id=binding['job_id'], completed_own_batches=1,
                        questions=128, trajectories=640, global_goal_complete=False)
                    publish(self.jobdir / 'progress.json', result)
                    publish(self.jobdir / 'sampler-exit.json', result)
                    self.returncode = 0
                    return self.returncode

            def stop(process):
                stopped.append(process.pid)
                if process.returncode is None:
                    process.returncode = -15

            pool.stop_group = stop

            def health(*_, **__):
                return io.BytesIO(json.dumps(dict(ready=True, pid=upstream_pid,
                    run_id=run_id, gpu_shards=4, logical_flat_shards=64,
                    encoder_dtype='float16', index_dtype='float16',
                    original_global_ids=True, endpoint=pool.URL)).encode())

            borrowed = types.SimpleNamespace(pid=9000, proof=dict(pid=9000, birth='123'), poll=lambda: None) if reuse else None
            with patch.object(CONSUMER, 'admit', return_value=(binding, [])), \
                    patch.object(CONSUMER, 'borrowed_relay', return_value=borrowed), \
                    patch.object(CONSUMER.subprocess, 'Popen', side_effect=FakeProcess), \
                    patch.object(CONSUMER.urllib.request, 'urlopen', side_effect=health), \
                    patch.object(CONSUMER.time, 'sleep', return_value=None), \
                    contextlib.redirect_stdout(io.StringIO()):
                CONSUMER.controller(pool, args)

            jobdir = next((base / 'pool/jobs/job-relay10-test').glob('attempt-*'))
            complete = read(jobdir / 'complete.json')
            self.assertEqual(complete['controller_status'], 'OWN_RELAY_CONSUMER_FINISHED')
            self.assertEqual((complete['completed_own_batches'], complete['questions'], complete['trajectories']),
                             (1, 128, 640))
            self.assertEqual(complete['sampler_returncode'], 0)
            self.assertTrue(complete['own_children_reaped'])
            self.assertFalse(complete['upstream_retriever_signalled'])
            self.assertFalse((jobdir / 'failure.json').exists())
            self.assertEqual([process.kind for process in processes], ['sampler'] if reuse else ['local-relay', 'sampler'])
            self.assertCountEqual(stopped, [process.pid for process in processes])
            self.assertNotIn(upstream_pid, stopped)
            self.assertNotIn(7000, stopped, 'The upstream relay is owned independently')
            self.assertNotIn(9000, stopped, 'Borrowed local relay must never be stopped')
            self.assertTrue((processes[-1].launch / 'model-initialized.json').is_file())
            self.assertEqual((jobdir / 'borrowed-local-relay.json').exists(), reuse)
            self.assertEqual(read(jobdir / 'retriever-ready.json')['pid'], upstream_pid)


if __name__ == '__main__':
    unittest.main()
