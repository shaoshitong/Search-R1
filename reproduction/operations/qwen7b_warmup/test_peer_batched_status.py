"""The read-only observer must survive progress replacement visibility races."""
import json,pathlib,tempfile,unittest
from .deploy_peer_batched_replay import STATUS

class OptionalStatusTests(unittest.TestCase):
    def functions(self,read):
        values={'json':json,'read':read}
        exec(STATUS.split("result=dict(status=")[0],values)
        return values
    def test_exists_then_removed_progress_is_unavailable_not_failure(self):
        with tempfile.TemporaryDirectory() as d:
            p=pathlib.Path(d)/'progress.json';p.write_text('{"trajectories":1}')
            def race(path):path.unlink();return json.loads(path.read_text())
            values=self.functions(race);self.assertIsNone(values['optional'](p))
    def test_partial_json_is_an_explicit_unavailable_observation(self):
        with tempfile.TemporaryDirectory() as d:
            p=pathlib.Path(d)/'progress.json';p.write_text('{')
            values=self.functions(lambda path:json.loads(path.read_text()))
            self.assertIsNone(values['optional'](p));self.assertEqual(values['read_errors'][0]['error'],'JSONDecodeError')
    def test_real_complete_progress_is_retained(self):
        with tempfile.TemporaryDirectory() as d:
            p=pathlib.Path(d)/'progress.json';p.write_text('{"trajectories":1024}')
            values=self.functions(lambda path:json.loads(path.read_text()))
            self.assertEqual(values['optional'](p),{'trajectories':1024})
            self.assertEqual(values['read_errors'],[])
if __name__=='__main__':unittest.main()
