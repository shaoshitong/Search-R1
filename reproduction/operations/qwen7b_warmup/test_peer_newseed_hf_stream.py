import copy
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
import base64
import shlex

from . import peer_newseed_hf_stream as m
from . import deploy_peer_newseed_hf as deploy


def plan():
    root=m.SOURCE_PARENT/'fixture_20261021'
    data=b'raw snapshot fixture'
    group=dict(group='part-0000',archive=str(root/'part-0000.tar.gz'),archive_bytes=len(data),
        archive_sha256=hashlib.sha256(data).hexdigest(),files=[dict(path='batch0/full.jsonl.gz',bytes=10,sha256='a'*64)],
        source_bytes=10,questions=1,trajectories=5)
    return root,dict(status='SEALED_CANONICAL_NEWSEED_PREFIX20_G5_SNAPSHOT_EXPORT_COMPLETE',
        source_run_id='70188b8e50fd4b45bc420ce58014f46e',seed=20261021,hidden_exported=False,
        all_archives_fresh_SHA_and_member_SHA_verified=True,
        source_node='h100-10',revision=m.REVISION,snapshot_id=root.name,
        actor_updates=0,fresh_all_source_files_SHA256=True,groups=[group],files=1,payload_bytes=10,questions=1,trajectories=5)


class StreamTests(unittest.TestCase):
    def test_archive_and_member_scope_rejects_escape_or_hidden(self):
        root,value=plan();self.assertEqual(m.validate_snapshot(value,root),value)
        for name in ['../secret','/absolute','hidden.bin','model.pt']:
            own=copy.deepcopy(value);own['groups'][0]['files'][0]['path']=name
            with self.assertRaises(ValueError):m.validate_snapshot(own,root)
        own=copy.deepcopy(value);own['groups'][0]['archive']='/tmp/other.tar.gz'
        with self.assertRaises(ValueError):m.validate_snapshot(own,root)
        own=copy.deepcopy(value);own['groups'][0]['source_bytes']=11
        with self.assertRaises(ValueError):m.validate_snapshot(own,root)

    def test_strict_ssh_existing_known_hosts_and_explicit_public_keys(self):
        c=dict(user='root',management_identity_file='/management-only/key',gateway_ip='10.0.0.18',targets={'h100-10':'10.0.0.10'})
        argv=m.source_args('hostname',c)
        self.assertIn('StrictHostKeyChecking=yes',argv)
        self.assertTrue(any(s.startswith('ProxyCommand=') for s in argv))
        self.assertNotIn('StrictHostKeyChecking=no',argv)
        c['trusted_target_host_keys']={'h100-10':['10.0.0.10 ssh-ed25519 PUBLIC_FIXTURE']}
        self.assertTrue(any(s.startswith('KnownHostsCommand=') for s in m.source_args('hostname',c)))

    def test_real_seekable_bounded_pipe_and_reopen(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'archive';payload=os.urandom(9*1024**2);p.write_bytes(payload)
            # This exercises the exact remote command through a local shell,
            # never a model/data transfer or a network action.
            f=m.RemoteFile(dict(archive=str(p),archive_bytes=len(payload)),argv=lambda cmd:['/bin/sh','-c',cmd])
            try:
                with self.assertRaisesRegex(ValueError,'Unbounded'):f.read()
                self.assertEqual(f.read(513),payload[:513])
                self.assertEqual(f.seek(1024),1024)
                self.assertEqual(f.read(20),payload[1024:1044])
                self.assertEqual(f.seek(-16,2),len(payload)-16)
                self.assertEqual(f.read(),payload[-16:])
                self.assertEqual(f.read(1),b'')
                self.assertEqual(f.seek(0),0)
                self.assertEqual(len(f.read(20*1024**2)),8*1024**2)
                with self.assertRaises(ValueError):f.seek(len(payload)+1)
            finally:f.close()

    def test_control_code_is_bounded_and_secret_only_on_stdin(self):
        root,value=plan();source=Path(m.__file__).read_bytes()
        inputs=dict(base=deploy.PARENT+'/'+root.name,source=str(root),snapshot=root.name,
            code_sha256=hashlib.sha256(source).hexdigest(),commit='a'*40,chunks=3,
            index=0,data=base64.b64encode(b'x'*4096).decode(),chunk_sha256='a'*64)
        for action in ('chunk','stage','start','status'):
            code=deploy.render(inputs,action);compile(code,'control-'+action,'exec')
            self.assertLess(len(('python3 -c '+shlex.quote(code)).encode()),16000)
        self.assertIn('secret=json.loads(sys.stdin.readline())',deploy.render(inputs,'start'))
        self.assertNotIn('shell=True',deploy.render(inputs,'start'))


if __name__=='__main__':unittest.main()
