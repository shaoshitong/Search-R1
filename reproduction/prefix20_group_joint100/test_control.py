"""CPU checks of the fresh experiment contract and code-only delta deployment."""
import copy
import hashlib
import io
import json
from pathlib import Path
import tarfile
import unittest
from unittest.mock import patch

from .control import archive_for_commit,deployment_script
from .launch import PREFIX,trainer_command,validate_config


def fixture():
    c=json.loads((Path(__file__).parent/'EXPERIMENT.json').read_text())
    c.update(base=PREFIX+'test',run_id='1'*32,source_inventory_sha256='2'*64)
    return c


def archive(contents):
    stream=io.BytesIO()
    with tarfile.open(fileobj=stream,mode='w:gz')as handle:
        for name,data in contents.items():
            item=tarfile.TarInfo(name);item.size=len(data);item.mode=0o644
            handle.addfile(item,io.BytesIO(data))
    return stream.getvalue()


class ControlTest(unittest.TestCase):
    def test_fresh_config_refuses_old_state_import_or_unconfirmed_data_schedule(self):
        c=fixture();self.assertIs(validate_config(c,Path(c['base'])),c)
        alternative=copy.deepcopy(c);alternative['data_schedule']=[1]*100
        with self.assertRaises(ValueError):validate_config(alternative,Path(c['base']))
        for key in ('resume_replay','source_checkpoint','source_adam_steps_per_rank'):
            invalid=dict(c,**{key:'older experiment state'})
            with self.assertRaises(ValueError):validate_config(invalid,Path(c['base']))
        script=deployment_script(c['base'],'a'*40,c,
            dict(archive=b'',files={},modes={},reuse_files={}))
        compile(script,'<isolated-once-deployment>','exec')

    def test_epoch71_uses_scheduled_tokens_inline_and_never_passes_old_checkpoint(self):
        c=fixture();base=Path(c['base'])
        command=trainer_command('/env/python',c,base,71,base/'epochs/epoch-0071/inputs',
                                Path('/remote/fixed-test/features'),'a'*64)
        self.assertEqual(command[command.index('--source-data-epoch')+1],'1')
        self.assertEqual(command[command.index('--total-optimizer-updates')+1],'10000')
        self.assertIn('--train-inputs',command)
        self.assertNotIn('--train-cache',command)
        self.assertNotIn('--source-checkpoint',command)
        self.assertEqual(command[command.index('--batch-semantics')+1],'average255_260_complete_G5')

    def test_delta_reuses_only_identical_blobs_and_retains_changed_and_new_source(self):
        old={'same.py':b'same','changed.py':b'old'}
        target={'same.py':b'same','changed.py':b'new','added.py':b'added'}
        def git(args):return archive(old if args[-1]=='b'*40 else target)
        with patch('subprocess.check_output',side_effect=git):
            built=archive_for_commit('a'*40,'b'*40)
        with tarfile.open(fileobj=io.BytesIO(built['archive']))as handle:
            delta={item.name:handle.extractfile(item).read()for item in handle if item.isfile()}
        self.assertEqual(set(delta),{'changed.py','added.py'})
        self.assertEqual(built['reuse_files'],{'same.py':hashlib.sha256(b'same').hexdigest()})
        combined={**{name:hashlib.sha256(value).hexdigest()for name,value in delta.items()},**built['reuse_files']}
        self.assertEqual(combined,built['files'])


if __name__=='__main__':unittest.main()
