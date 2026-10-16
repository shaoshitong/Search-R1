import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from reproduction.common8 import native_allocation as module


class NativeInventoryTests(unittest.TestCase):
    def inventory(self):
        rows=[[str(i),'GPU-'+str(i),'80000'] for i in range(4)]
        xml='<nvidia_smi_log>'+''.join('<gpu><uuid>GPU-%d</uuid><minor_number>%d</minor_number></gpu>'%(i,i+4) for i in reversed(range(4)))+'</nvidia_smi_log>'
        return rows,xml

    def test_xml_uuid_join_rejects_missing_or_ambiguous_physical_mapping(self):
        rows,xml=self.inventory()
        self.assertEqual(module.add_gpu_minor_numbers(rows,xml),[row+[str(i+4)] for i,row in enumerate(rows)])
        for invalid in (xml.replace('GPU-3','GPU-2'),xml.replace('GPU-3','GPU-x'),xml.replace('>7<','>4<'),xml.replace('>7<','>N/A<')):
            with self.assertRaises(AssertionError):module.add_gpu_minor_numbers(rows,invalid)

    def test_full_native_four_admission_uses_supported_csv_and_preserves_xml(self):
        rows,xml=self.inventory();calls=[];node='ACD1-32'
        detail='JobId=123 JobState=RUNNING NodeList='+node+' NumNodes=1 TresPerNode=gres/gpu:4\n Nodes='+node+' CPU_IDs=0-47 GRES=gpu:4(IDX:4-7)\n'
        def query(args,**kwargs):
            calls.append(args)
            if args[0]=='scontrol':return detail
            if args[0]=='squeue':return '123\n456\n'
            if args==['nvidia-smi','-q','-x']:return xml
            if '--query-compute-apps=gpu_uuid,pid' in args:return 'GPU-foreign, 987\n'
            if '--query-gpu=index,uuid,memory.total' in args:return '\n'.join(','.join(row) for row in rows)
            if '--query-gpu=index,uuid,memory.total,memory.used,memory.free' in args:
                return '\n'.join(','.join(row+['0','80000']) for row in rows)
            raise AssertionError('Unsupported query '+repr(args))
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory).resolve();(base/'logs').mkdir()
            (base/'logs/run-manifest.json').write_text(json.dumps(dict(profile='common8',run_id='run',completed_updates=0,training_world_size=4,execution_site='hpc3')))
            torch=SimpleNamespace(cuda=SimpleNamespace(device_count=lambda:4))
            with patch.dict('sys.modules',{'torch':torch}),patch.dict('os.environ',SLURM_JOB_ID='123',CUDA_VISIBLE_DEVICES='0,1,2,3',SEARCH_G1_TRAINING_WORLD_SIZE='4'),patch.object(module.socket,'gethostname',return_value=node),patch.object(module.subprocess,'check_output',side_effect=query),patch.object(module,'cuda_uuids',return_value=[row[1] for row in rows]):
                proof=module.admit(base)
            self.assertTrue(proof['accepted'])
            self.assertEqual(proof['actual_allocated_indices'],[4,5,6,7])
            self.assertEqual(proof['actual_nvml_indices'],[0,1,2,3])
            self.assertEqual((base/'logs/native-gpu-inventory.xml').read_text(),xml)
            self.assertFalse(any('minor_number' in arg for args in calls for arg in args if arg.startswith('--query-gpu=')))


if __name__=='__main__':unittest.main()
