import shlex,unittest
from . import deploy_peer_newseed_receive as m


class DeployTests(unittest.TestCase):
    def test_master_enabled_once_and_reuse_cannot_fallback(self):
        args=m.master_args('hkust2','/tmp/own.sock','ssh -W %h:%p hkust1')
        self.assertNotIn('-M',args)
        self.assertEqual(args.count('-oControlMaster=yes'),1)
        self.assertEqual(args[-1],'hkust2')
        client=m.master_client('hkust2','/tmp/own.sock')
        self.assertIn('-oControlMaster=no',client)
        self.assertIn('-oProxyCommand=false',client)

    def test_controls_bounded_and_secret_only_stdin(self):
        values=dict(base='/data/shaoshitong/qwen7b-peer-newseed-to-hpc3-20261021/'+m.SNAPSHOT,
            snapshot=m.SNAPSHOT,code='/data/code/peer_newseed_receive.py',code_sha256='a'*64,
            source_commit='b'*40,prefix='peer-newseed-to-hpc3-20261021/'+m.SNAPSHOT,catalog_commit='c'*40,hf_proxy=None)
        for action in ('start','status'):
            source=m.render(values,action);compile(source,'receiver-'+action,'exec')
            self.assertLess(len(('python3 -c '+shlex.quote(source)).encode()),16000)
        source=m.render(values,'start')
        self.assertIn('secret=json.loads(sys.stdin.readline())',source)
        self.assertIn("assert not (b/'launch-intent.json').exists()",source)
        self.assertIn("EXISTING_RECEIVER_LAUNCH_NOT_REPLAYED",source)
        self.assertNotIn('shell=True',source)

    def test_metadata_only_no_inherited_or_Mac_proxy(self):
        compile(m.PREFLIGHT,'metadata-preflight','exec')
        self.assertIn('session.trust_env=False',m.PREFLIGHT)
        self.assertIn('17894',m.PREFLIGHT);self.assertIn('17897',m.PREFLIGHT)
        self.assertNotIn('18092',m.PREFLIGHT);self.assertIn('stream=True',m.PREFLIGHT)
        self.assertIn("response.status_code!=206",m.PREFLIGHT);self.assertIn("Content-Range",m.PREFLIGHT)
        self.assertIn("value['range_bytes_read']==limit",m.PREFLIGHT)


if __name__=='__main__':unittest.main()
