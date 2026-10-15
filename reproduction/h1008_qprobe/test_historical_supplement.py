"""Small CPU fixtures only; no actual backup preparation or start."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from reproduction.h1008_qprobe import historical_supplement as s

class HistoricalSupplementTests(unittest.TestCase):
    def setUp(self):
        self.plan=json.loads(Path(s.__file__).with_name('historical_supplement_plan.json').read_text())

    def test_exact_three_file_plan_and_pair_records(self):
        rows=s.validate_plan(self.plan)
        records=[dict(path=r['source_path'],full_read_sha256=r['expected_sha256'],bytes=r['bytes']) for r in rows]
        pair=dict(run_id=s.RUN,step=5,status='QPROBE_COMPLETED_PAIR_FULL_SHA_CPU_STATES_PASS',historical_artifacts=dict(development=dict(before=records[0],after5=records[1]),reward_archives=[records[2]]))
        s.validate_pair_rows(pair,rows)
        pair['historical_artifacts']['development']['after5']['full_read_sha256']='0'*64
        with self.assertRaises(ValueError):s.validate_pair_rows(pair,rows)

    def test_parent_rate_namespace_duplicate_and_payload_drift_rejected(self):
        for key,value in [('parent_inventory_sha256','0'*64),('rate_bytes_per_second',2*s.RATE),('recommended_target_root','/tmp/wrong')]:
            changed=copy.deepcopy(self.plan);changed[key]=value
            with self.assertRaises(ValueError):s.validate_plan(changed)
        changed=copy.deepcopy(self.plan);changed['files'][0]['payload_path']='../escape'
        with self.assertRaises(ValueError):s.validate_plan(changed)

    def test_parent_success_and_bound_birth_are_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'checkpoint-closure').mkdir()
            result=dict(status='INDEPENDENT_INVENTORY_FULL_SHA_PASS',run_id=s.RUN,step=5,inventory_sha256=s.PARENT_SHA,files=1122,bytes=137694774279,final_directory=str(root/'checkpoint-closure'))
            intent=dict(run_id=s.RUN,inventory_sha256=s.PARENT_SHA,worker_source_sha256=s.WORKER_SHA,rate_bytes_per_second=s.RATE,ordinary_reserve_bytes=s.RESERVE)
            for name,value in [('RESULT.json',result),('INTENT.json',intent),('LAUNCH.json',dict(pid=2245796,start_ticks='1913194447'))]:(root/name).write_text(json.dumps(value))
            with patch.object(s,'sha',side_effect=lambda p:s.WORKER_SHA if Path(p).name=='copy-worker.py' else s.PARENT_SHA):
                self.assertTrue(s.parent_finished(root)['writer_birth_not_live'])
                result['status']='NOT_COMPLETE';(root/'RESULT.json').write_text(json.dumps(result))
                with self.assertRaises(ValueError):s.parent_finished(root)

    def test_bound_live_parent_refuses_without_launch(self):
        fields=['S']+['0']*18+['1913194447']
        original=Path.read_text
        def text(path,*args,**kwargs):
            if str(path)=='/proc/2245796/stat':return '2245796 (fixture) '+' '.join(fields)
            return original(path,*args,**kwargs)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            values={'RESULT.json':dict(status='INDEPENDENT_INVENTORY_FULL_SHA_PASS',run_id=s.RUN,step=5,inventory_sha256=s.PARENT_SHA,files=1122,bytes=137694774279,final_directory=str(root/'checkpoint-closure')),'INTENT.json':dict(run_id=s.RUN,inventory_sha256=s.PARENT_SHA,worker_source_sha256=s.WORKER_SHA,rate_bytes_per_second=s.RATE,ordinary_reserve_bytes=s.RESERVE),'LAUNCH.json':dict(pid=2245796,start_ticks='1913194447')}
            for name,value in values.items():(root/name).write_text(json.dumps(value))
            with patch.object(s,'sha',side_effect=lambda p:s.WORKER_SHA if Path(p).name=='copy-worker.py' else s.PARENT_SHA),patch.object(Path,'read_text',text):
                with self.assertRaisesRegex(ValueError,'STILL_LIVE'):s.parent_finished(root)

    def test_existing_supplement_target_refuses_before_parent_query(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan_path=Path(tmp)/'plan.json';plan_path.write_text(json.dumps(self.plan))
            original=Path.exists
            def exists(path):return True if str(path)==self.plan['recommended_target_root'] else original(path)
            with patch.object(s.socket,'gethostname',return_value='h100-10'),patch.object(Path,'exists',exists),patch.object(s,'parent_finished') as parent:
                with self.assertRaisesRegex(ValueError,'OCCUPIED'):s.start_target(plan_path,'a'*40)
                parent.assert_not_called()

if __name__=='__main__':unittest.main()
