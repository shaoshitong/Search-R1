"""Small CPU coverage of explicit new-service plumbing; no actual submissions."""
from contextlib import contextmanager
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from reproduction.operations.easy7b_20261021 import (
    fa3_transition, fa3_lean_transition, publish_retrieval_external_receipt as publisher,
    submit_private_retrieval_service_hpc as service_submit,
)


class ExternalRetrievalTransitionTests(unittest.TestCase):
    def test_service_defaults_and_explicit_seven_day_producer_arguments(self):
        for production in (False, True):
            with self.subTest(production=production), tempfile.TemporaryDirectory() as directory:
                base = Path(directory)
                request = dict(base=str(base), commit='a'*40, index='/old/index', corpus='/old/corpus')
                if production:
                    request.update(time_limit='7-00:00:00', stage_parquet='/source/data/stage.parquet',
                                   training_run_id='b'*32, job_name='easy7b-retrieval-batch16')
                completed = SimpleNamespace(returncode=0, stdout='9999\n', stderr='')
                with patch.object(service_submit.sys, 'stdin', io.StringIO(json.dumps(request))), \
                        patch.object(service_submit.subprocess, 'run', return_value=completed) as run, \
                        patch('sys.stdout', io.StringIO()):
                    service_submit.main()
                command = run.call_args.args[0]
                self.assertIn('--gres=gpu:2', command)
                self.assertIn('--time=' + ('7-00:00:00' if production else '04:00:00'), command)
                wrapped = command[-1]
                self.assertIn('--faiss-batch-size 16 --faiss-max-wait-ms 2', wrapped)
                self.assertEqual('--stage-parquet /source/data/stage.parquet' in wrapped, production)
                self.assertEqual('--training-run-id ' + 'b'*32 in wrapped, production)
                self.assertEqual((base / 'job.id').read_text(), '9999\n')
                with patch.object(service_submit.sys, 'stdin', io.StringIO(json.dumps(request))), \
                        patch.object(service_submit.subprocess, 'run') as duplicate, patch('sys.stdout', io.StringIO()):
                    service_submit.main()
                duplicate.assert_not_called()

    def test_receipt_preserves_true_ready_and_is_once_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'logs').mkdir()
            ready = dict(ready=True, source_commit='a'*40, service_job_id='9999',
                endpoint='http://node:18026/retrieve', request_rebatching_enabled=True,
                singleton_e5=True, faiss_batch_size=16, faiss_max_wait_ms=2)
            path = root / 'logs/service-ready.json'
            path.write_text(json.dumps(ready))
            output = root / 'preparation/external-retriever-receipt.json'
            first = publisher.publish_receipt(root, output)
            self.assertEqual(first, publisher.publish_receipt(root, output))
            self.assertEqual(json.loads(path.read_text()), ready)
            self.assertEqual(set(json.loads(output.read_text())), {
                'service_ready_path', 'service_ready_sha256', 'source_commit', 'service_job_id', 'endpoint'})

    def test_prepare_defaults_old_binding_or_accepts_explicit_actual_new_job(self):
        for explicit in (False, True):
            with self.subTest(explicit=explicit), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                target = root / 'target'; (target / 'preparation').mkdir(parents=True)
                old, new = root / 'old.json', root / 'new.json'
                old.write_text(json.dumps(dict(service_job_id='745198')))
                new.write_text(json.dumps(dict(service_job_id='9999')))
                kwargs = dict(external_receipt=str(new), expected_service_job='9999') if explicit else {}
                binding = lambda path: json.loads(Path(path).read_text())
                with patch.object(fa3_transition, '_target', return_value=target), \
                        patch.object(fa3_transition, '_source', return_value=dict(external_retriever=dict(receipt_path=str(old)))), \
                        patch.object(fa3_transition.control, 'bootstrap', return_value=dict(environment={})), \
                        patch.object(fa3_transition, '_consumers', return_value={}), \
                        patch('reproduction.easy2560_joint4.retrieval_proxy.external_binding', side_effect=binding), \
                        patch('reproduction.easy2560_joint4.flash3_runtime.prepare', return_value={}):
                    result = fa3_transition.prepare(target, 'c'*40, **kwargs)
                    self.assertEqual(result['external_retriever_service_job'], '9999' if explicit else '745198')
                    self.assertEqual(result, fa3_transition.prepare(target, 'c'*40, **kwargs))
                copied = json.loads((target / 'preparation/external-retriever-receipt.json').read_text())
                self.assertEqual(copied['service_job_id'], '9999' if explicit else '745198')

    def test_lean_prepare_cli_forwards_new_receipt_without_changing_submit_api(self):
        with tempfile.TemporaryDirectory() as directory:
            auth = Path(directory) / 'auth.json'; auth.write_text('{}')
            from unittest.mock import Mock
            bound = SimpleNamespace(prepare=Mock(return_value={}), submit=Mock(return_value={}))
            @contextmanager
            def binding(**kwargs):
                yield bound
            args = ['prepare', '--commit', 'a'*40, '--authorization-json', str(auth),
                    '--external-receipt', '/new/external.json', '--expected-service-job', '9999']
            with patch.object(fa3_lean_transition, 'source_binding', binding), patch('sys.stdout', io.StringIO()):
                fa3_lean_transition.main(args)
            bound.prepare.assert_called_once_with(fa3_lean_transition.TARGET_BASE, 'a'*40,
                external_receipt='/new/external.json', expected_service_job='9999')
            bound.submit.assert_not_called()


if __name__ == '__main__':
    unittest.main()
