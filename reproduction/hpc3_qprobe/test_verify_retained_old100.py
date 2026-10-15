import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch

from reproduction.hpc3_qprobe import verify_retained_old100 as v


class Old100Tests(unittest.TestCase):
    def metadata(self):
        p=v.reserve.plan();checkpoint=v.OLD_BASE/'checkpoints/search-g1-common8/actor/global_step_100'
        files=[dict(path=str(path),inode=1,bytes=3,mtime_ns=2,full_read_sha256='a'*64) for path in
            (checkpoint/'model.bin',Path(p['groups'][0]['checkpoint'])/'complete.json',
             Path(p['groups'][1]['checkpoint'])/'complete.json')]
        receipt=dict(status='QPROBE_COMPLETED_PAIR_FULL_SHA_CPU_STATES_PASS',run_id=p['old_run_id'],step=100,
                     world_size=4,source_base=str(v.OLD_BASE),checkpoint=str(checkpoint),files=files)
        transport=dict(status='QPROBE_PAIR_SOURCE_ARCHIVE_TARGET_FULL_SHA_PASS',run_id=p['old_run_id'],step=100,
                       world_size=4,files=[dict(source_path=r['path'],bytes=r['bytes'],sha256=r['full_read_sha256']) for r in files])
        return p,transport,receipt

    def test_exact_retained_source_set_and_two_metadata_dependencies(self):
        p,t,r=self.metadata();rows,keep=v.expected_metadata(p,t,r)
        self.assertEqual(len(rows),3);self.assertEqual(len(keep),2)
        self.assertTrue(all(Path(x).name=='complete.json' for x in keep))

    def test_wrong_world_source_or_omitted_file_refuses(self):
        for change in ('world','run','missing','sha'):
            p,t,r=self.metadata()
            if change=='world':r['world_size']=8
            elif change=='run':r['run_id']=p['new_run_id']
            elif change=='missing':t['files'].pop()
            else:t['files'][0]['sha256']='b'*64
            with self.subTest(change=change):
                with self.assertRaises(ValueError):v.expected_metadata(p,t,r)

    def test_bulk_alias_of_prospective_retirement_refuses(self):
        p,t,r=self.metadata();row=copy.deepcopy(r['files'][1]);row['path']=str(Path(p['groups'][0]['checkpoint'])/'rank-0.pt')
        r['files'].append(row);t['files'].append(dict(source_path=row['path'],bytes=3,sha256='a'*64))
        with self.assertRaises(ValueError):v.expected_metadata(p,t,r)

    def test_conflicting_or_current_run_source_records_refuse(self):
        _,_,r=self.metadata();rows=copy.deepcopy(r['files']);conflict=dict(rows[0],bytes=4);rows.append(conflict)
        with self.assertRaises(ValueError):v.unique_records(rows)
        with self.assertRaises(ValueError):v.unique_records([dict(rows[0],path=str(v.reserve.BASE/'checkpoint'))])

    def test_failed_binding_still_waits_for_direct_child(self):
        parent=dict(pid=101,start_ticks=1001,sid=101,uid=14090);childrow=dict(pid=102,start_ticks=1002,sid=101,uid=14090)
        child=Mock(pid=102,stdin=io.BytesIO());child.wait.return_value=0
        publish=v.reserve.publish
        def injected(path,value,replace=False):
            if Path(path).name=='worker-process.json':raise OSError('injected publication error')
            return publish(path,value,replace)
        with tempfile.TemporaryDirectory() as tmp:
            control=Path(tmp)
            with patch.object(v,'DIRECTORY',control),patch.object(v,'code_guard',return_value=(None,control)),\
                 patch.object(v.reserve,'read_owned',return_value=dict(source_commit='a'*40,actual_process=parent)),\
                 patch.object(v.reserve,'birth',side_effect=lambda pid:childrow if pid==102 else parent),\
                 patch.object(v.reserve,'publish',side_effect=injected),patch.object(v.os,'getsid',return_value=101),\
                 patch.object(v.subprocess,'Popen',return_value=child):
                self.assertEqual(v.supervise('a'*40),1)
            child.wait.assert_called_once_with()
            result=json.loads((control/'result.json').read_text());self.assertEqual(result['actual_child_exit_code'],0)
            self.assertIn('injected publication error',result['error'])


if __name__=='__main__':unittest.main()
