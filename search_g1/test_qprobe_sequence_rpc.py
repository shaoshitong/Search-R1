import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from search_g1.qprobe_sequence_rpc import RemoteSequenceQProbe,execute_probe,worker_transport
from search_g1.qprobe_model import OnlineQProbe


class SequenceRPCTests(unittest.TestCase):
    def proxy(self):
        import search_g1.test_qprobe_sequence_model as helpers
        helper=helpers.OnlineSequenceTests();head=helper.probe();calls=[]
        cache=SimpleNamespace(path=Path('/remote/features'),spec_sha='c'*64)
        def transport(op,payload):
            calls.append(op)
            if op=='init':return dict(feature_dim=8,feature_spec_sha='c'*64,config=vars(head.config),
                train_micro=2,predict_micro=2,initialization=head.initialization,version=head.version,
                completed_outer=0,update_count=0,actor_snapshot_id=None,worker_rank=7,world_size=8,
                actual_probe_device='cuda:7',gpu_uuid='synthetic-test-gpu7')
            return execute_probe(head,op,payload)
        proxy=RemoteSequenceQProbe(cache,{},transport)
        return helper,head,proxy,calls
    def test_only_existing_rank7_prefixed_method_is_called_without_new_ray_actor(self):
        events=[]
        class Method:
            def remote(self,*args):events.append(args);return 'result-ref'
        worker=SimpleNamespace(_ray_method_signatures={'actor_rollout_qprobe_operation':None},actor_rollout_qprobe_operation=Method())
        group=SimpleNamespace(world_size=8,workers=[object() for _ in range(7)]+[worker])
        fake=SimpleNamespace(get=lambda ref:'actual-result')
        with patch.dict('sys.modules',{'ray':fake}):
            call=worker_transport(group);self.assertEqual(call('state_dict',{}),'actual-result')
        self.assertEqual(events,[('state_dict',{})])
    def test_remote_fit_checkpoint_rollback_and_pairing_use_same_local_protocol(self):
        helper,head,proxy,calls=self.proxy();features=helper.features()
        a=proxy.predict(features);b=head.predict(features)
        torch.testing.assert_close(a['probabilities'],b['probabilities'],rtol=0,atol=0)
        helper.update(proxy);before=proxy.state_dict();version=proxy.version
        helper.update(proxy,2);proxy.load_state_dict(before);self.assertEqual(proxy.version,version)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'head.pt';proof=proxy.save(path,actor_snapshot_id='actor1')
            restored=OnlineQProbe.load(path,expected_actor_snapshot_id='actor1',expected_checkpoint_sha=proof['sha256'])
            self.assertEqual(restored.version,proxy.version);self.assertEqual(head.actor_snapshot_id,'actor1')
        self.assertIn('update_completed_outer',calls);self.assertIn('bind_actor',calls)
    def test_unknown_operation_and_gpu_owner_relocation_rejected(self):
        helper,head,proxy,calls=self.proxy()
        with self.assertRaisesRegex(ValueError,'OPERATION_INVALID'):execute_probe(head,'arbitrary',{})
        with self.assertRaisesRegex(ValueError,'RELOCATION'):proxy.set_device('cuda:0')
        proxy.set_device('allocated_rank7')
    def test_actual_worker_source_registers_endpoint_on_original_class(self):
        import ast
        from pathlib import Path
        tree=ast.parse((Path(__file__).resolve().parents[1]/'verl/workers/fsdp_workers.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ActorRolloutRefWorker')
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='qprobe_operation')
        self.assertTrue(method.decorator_list)
        self.assertIn('worker_operation',ast.unparse(method))

if __name__=='__main__':unittest.main()
