"""Once-only launch and actual output/exit evidence, without external work."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from reproduction.hpc3_qprobe import independent_goal_control719225 as control


class IndependentFinalControlTests(unittest.TestCase):
    def test_missing200_creates_no_scope_claim_or_process(self):
        with tempfile.TemporaryDirectory() as folder:
            directory=Path(folder).resolve()/'absent-on-purpose'
            with patch.object(control,'DIRECTORY',directory),patch.object(control,'caps'),\
                    patch.object(control.reader,'early_history',side_effect=ValueError('missing actual200')),\
                    patch.object(control.subprocess,'Popen') as launch,\
                    patch.object(control.ctl,'source') as source,patch.object(control.ctl,'publish') as publish:
                with self.assertRaisesRegex(ValueError,'missing actual200'):control.start('a'*40)
                self.assertFalse(directory.exists());launch.assert_not_called();source.assert_not_called();publish.assert_not_called()

    def test_existing_unknown_scope_is_inspected_never_replayed(self):
        with tempfile.TemporaryDirectory() as folder:
            directory=Path(folder).resolve()
            with patch.object(control,'DIRECTORY',directory),patch.object(control,'caps'),\
                    patch.object(control,'inspect',return_value={'status':'UNKNOWN_FINAL_CONSUMER_NO_REPLAY'}) as inspect,\
                    patch.object(control,'ready') as ready,patch.object(control.subprocess,'Popen') as launch:
                self.assertEqual(control.start('a'*40)['status'],'UNKNOWN_FINAL_CONSUMER_NO_REPLAY')
                inspect.assert_called_once_with('a'*40);ready.assert_not_called();launch.assert_not_called()

    def fixture(self,directory):
        commit='a'*40;source=dict(commit=commit,publication_sha256='b'*64)
        process=dict(pid=100,start_ticks=9,sid=100,uid=14090)
        child=dict(pid=102,start_ticks=10,sid=100,uid=14090)
        claim=dict(source=source,created_unix=1)
        proof=dict(status=control.reader.STATUS,run_id=control.reader.RUN,source=source,
            completed_outer=200,world_size=4,actual_native_two_readbacks_reconsumed=True,
            update201_performed=False,goal_complete=False,actor_model_loaded=False,
            receipt_or_cache_or_claim_writes=0,training_or_helper_or_native_job_submissions=0,
            generated_trajectories=0,new_actor_head_or_scheduler_updates=0)
        for name,value in [('claim',claim),('process',process),('child',child)]:
            (directory/(name+'.json')).write_text(json.dumps(value))
        (directory/'consumer.stdout').write_text(json.dumps(proof));(directory/'consumer.stderr').write_bytes(b'')
        outputs={name:dict(path=str(directory/('consumer.'+name)),
            bytes=(directory/('consumer.'+name)).stat().st_size,
            sha256=control.ctl.digest(directory/('consumer.'+name))) for name in ('stdout','stderr')}
        result=dict(source=source,claim_sha256=control.ctl.digest(directory/'claim.json'),process=process,child=child,
            started_unix=2,completed_unix=3,actual_child_exit_code=0,supervisor_exit_code=0,error=None,**outputs)
        (directory/'result.json').write_text(json.dumps(result))
        return commit,result

    def test_real_output_bytes_exit_and_child_terminal_are_required(self):
        with tempfile.TemporaryDirectory() as folder:
            directory=Path(folder).resolve();commit,result=self.fixture(directory)
            reader=lambda p:json.loads(Path(p).read_text())
            with patch.object(control,'DIRECTORY',directory),patch.object(control.ctl,'read',side_effect=reader),\
                    patch.object(control.ctl,'source',return_value=(directory,result['source'])),\
                    patch.object(control.ctl,'live',return_value=False) as live:
                accepted=control.consume(commit)
                self.assertEqual(accepted['actual_child_exit_code'],0)
                self.assertFalse(accepted['goal_complete'])
                bad=copy.deepcopy(result);bad['actual_child_exit_code']=1
                (directory/'result.json').write_text(json.dumps(bad))
                with self.assertRaisesRegex(ValueError,'held final consumer wait'):control.consume(commit)
                (directory/'result.json').write_text(json.dumps(result))
                live.return_value=True
                with self.assertRaisesRegex(ValueError,'held final consumer wait'):control.consume(commit)
                live.return_value=False
                # The JSON still says PASS; changed physical stdout must fail.
                p=directory/'consumer.stdout';p.write_bytes(p.read_bytes()+b' ')
                with self.assertRaisesRegex(ValueError,'output changed'):control.consume(commit)

    def test_summary_pass_with_new_work_or_logical201_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            directory=Path(folder).resolve();commit,_=self.fixture(directory)
            proof=json.loads((directory/'consumer.stdout').read_text())
            for key,value in [('update201_performed',True),('actor_model_loaded',True),
                    ('training_or_helper_or_native_job_submissions',1),('new_actor_head_or_scheduler_updates',1)]:
                bad=copy.deepcopy(proof);bad[key]=value
                with self.assertRaisesRegex(ValueError,'stdout is not the full pure200'):control.accept_output(bad,commit)

    def test_missing_or_null_kernel_birth_does_not_count_as_terminal(self):
        with tempfile.TemporaryDirectory() as folder:
            directory=Path(folder).resolve();commit,result=self.fixture(directory)
            reader=lambda p:json.loads(Path(p).read_text())
            with patch.object(control,'DIRECTORY',directory),patch.object(control.ctl,'read',side_effect=reader),\
                    patch.object(control.ctl,'source',return_value=(directory,result['source'])),\
                    patch.object(control.ctl,'live',return_value=False):
                for target in ('process','child'):
                    for missing in (False,True):
                        bad=copy.deepcopy(result)
                        if missing:bad[target].pop('start_ticks')
                        else:bad[target]['start_ticks']=None
                        (directory/(target+'.json')).write_text(json.dumps(bad[target]))
                        (directory/'result.json').write_text(json.dumps(bad))
                        with self.assertRaisesRegex(ValueError,'held final consumer wait'):control.consume(commit)
                        (directory/(target+'.json')).write_text(json.dumps(result[target]))


if __name__=='__main__':unittest.main()
