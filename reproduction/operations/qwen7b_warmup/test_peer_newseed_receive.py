import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import tarfile
import unittest
from unittest.mock import patch

from . import peer_newseed_receive as m


def fixture():
    snapshot='snapshot_20261021'; data=b'own raw fixture\n'; member=snapshot+'/group-00000/source/full.jsonl'
    record=dict(group='group-00000',archive='/source/'+snapshot+'/part-00000.tar.gz',archive_bytes=10240,
        archive_sha256='b'*64,files=[dict(path=member,bytes=len(data),sha256=hashlib.sha256(data).hexdigest())],
        source_bytes=len(data),questions=1,trajectories=5)
    source=dict(snapshot_id=snapshot,source_node='h100-10',revision=m.REVISION,actor_updates=0,
        fresh_all_source_files_SHA256=True,groups=[copy.deepcopy(record)],files=1,payload_bytes=len(data),questions=1,trajectories=5)
    receipt=dict(record,status='PEER_SOURCE_PRIVATE_HF_ARCHIVE_SHA_ACCEPTED',repo=m.REPO,repo_type='dataset',private=True,
        hf_archive_path=m.PREFIX+snapshot+'/archives/part-00000.tar.gz',source_revision=m.REVISION,snapshot_id=snapshot)
    catalog=dict(status='ALL_PEER_NEWSEED_SOURCE_PRIVATE_HF_ARCHIVES_SHA_ACCEPTED_NOT_HPC3_COMPLETE',
        snapshot_id=snapshot,repo=m.REPO,repo_type='dataset',private=True,prefix=m.PREFIX+snapshot,
        source_snapshot=source,groups=[receipt],Mac_data_transit=False)
    return catalog,data


def write_tar(path,members):
    with tarfile.open(str(path),'w') as archive:
        for name,data,kind in members:
            info=tarfile.TarInfo(name); info.size=len(data)
            if kind=='symlink': info.type=tarfile.SYMTYPE;info.linkname='/tmp/escape';info.size=0
            if kind=='hardlink': info.type=tarfile.LNKTYPE;info.linkname='other';info.size=0
            archive.addfile(info,io.BytesIO(data) if info.isfile() else None)


class ReceiveTests(unittest.TestCase):
    def test_catalog_source_binding_and_LFS_exact(self):
        catalog,_=fixture();snapshot=catalog['snapshot_id']
        self.assertIs(m.validate_catalog(catalog,snapshot,m.PREFIX+snapshot),catalog)
        g=catalog['groups'][0];inventory=[dict(path=g['hf_archive_path'],size=g['archive_bytes'],lfs=dict(oid=g['archive_sha256']))]
        m.verify_inventory(catalog,inventory)
        for key,value in [('archive_bytes',99),('archive_sha256','c'*64),('source_revision','d'*40)]:
            bad=copy.deepcopy(catalog);bad['groups'][0][key]=value
            with self.assertRaises(ValueError):m.validate_catalog(bad,snapshot,m.PREFIX+snapshot)
        inventory[0]['lfs']['oid']='c'*64
        with self.assertRaises(ValueError):m.verify_inventory(catalog,inventory)
        with self.assertRaises(ValueError):m.validate_catalog(catalog,snapshot,'unrelated')

    def test_catalog_global_member_duplicates_and_own_G5_count_rejected(self):
        c,_=fixture(); bad=copy.deepcopy(c);bad['groups'][0]['files']*=2;bad['source_snapshot']['groups'][0]['files']*=2
        with self.assertRaises(ValueError):m.validate_catalog(bad,c['snapshot_id'],c['prefix'])
        bad=copy.deepcopy(c);bad['groups'][0]['trajectories']=4;bad['source_snapshot']['groups'][0]['trajectories']=4
        with self.assertRaises(ValueError):m.validate_catalog(bad,c['snapshot_id'],c['prefix'])
        self.assertEqual(m.group_name(0),'group-00000')

    def test_safe_exact_tar_and_real_fresh_readback(self):
        catalog,data=fixture();record=catalog['groups'][0];name=record['files'][0]['path']
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp).resolve();archive=base/'named.tar.gz';root=base/'raw'
            # Source exporter currently uses .tar; the reader also accepts a
            # historical .tar.gz suffix on an uncompressed deterministic tar.
            write_tar(archive,[(name,data,'regular')])
            self.assertEqual(m.safe_extract(archive,root,record),1)
            self.assertEqual((root/name).read_bytes(),data)
            self.assertEqual(m.safe_extract(archive,root,record),1)
            (root/name).chmod(0o600);(root/name).write_bytes(b'x'*len(data))
            with self.assertRaises(ValueError):m.safe_extract(archive,root,record)

    def test_tar_escape_links_duplicates_extra_missing_rejected(self):
        catalog,data=fixture();record=catalog['groups'][0];name=record['files'][0]['path']
        variants=[[(name,data,'symlink')],[(name,data,'hardlink')],[(name,data,'regular'),(name,data,'regular')],
            [('../escape',data,'regular')],[(name,data,'regular'),('foreign',data,'regular')],[]]
        for members in variants:
            with self.subTest(members=members),tempfile.TemporaryDirectory() as temp:
                archive=Path(temp).resolve()/'a.tar';write_tar(archive,members)
                with self.assertRaises(ValueError):m.safe_extract(archive,Path(temp).resolve()/'raw',record)
        for name_ in ('/absolute','../escape','a/../b','a//b','a\\b','model.pt','model.safetensors'):
            with self.assertRaises(ValueError):m.safe_member(name_)

    def test_symlink_parent_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp).resolve();(base/'foreign').mkdir();(base/'link').symlink_to(base/'foreign')
            with self.assertRaises(ValueError):m.secure_dir(base/'link'/'target')

    def test_exact_target_scope_catalog_binding_and_immutable_conflict(self):
        c,_=fixture();raw=json.dumps(c);b=dict(prefix=c['prefix'],catalog_commit='a'*40,
            catalog_bytes=raw,catalog_sha256=hashlib.sha256(raw.encode()).hexdigest())
        with tempfile.TemporaryDirectory() as temp:
            parent=Path(temp).resolve();base=parent/c['snapshot_id'];m.setup(base,parent,c,b);m.setup(base,parent,c,b)
            bad=dict(b,catalog_commit='b'*40)
            with self.assertRaises(ValueError):m.setup(base,parent,c,bad)
            with self.assertRaises(ValueError):m.setup(base/'other',parent,c,b)

    def test_real_target_accept_and_global_full_readback(self):
        catalog,data=fixture();snapshot=catalog['snapshot_id'];record=catalog['groups'][0]
        with tempfile.TemporaryDirectory() as temp:
            parent=Path(temp).resolve();base=parent/snapshot;m.secure_dir(base/'archives')
            archive,partial=m.paths(base,record);write_tar(partial,[(record['files'][0]['path'],data,'regular')])
            for own in (record,catalog['source_snapshot']['groups'][0]):
                own['archive_bytes']=partial.stat().st_size;own['archive_sha256']=m.sha(partial)
            raw=json.dumps(catalog);binding=dict(prefix=catalog['prefix'],catalog_commit='a'*40,
                catalog_bytes=raw,catalog_sha256=hashlib.sha256(raw.encode()).hexdigest())
            with patch.object(m,'TARGET',parent):
                m.target_init(base,dict(catalog=catalog,binding=binding))
                result=m.target_accept(base,dict(group=record['group']))
                self.assertEqual(result['files'],1);self.assertTrue(archive.exists());self.assertFalse(partial.exists())
                self.assertEqual(m.target_finalize(base,{})['status'],m.STATUS)
                self.assertEqual(m.target_accept(base,dict(group=record['group']))['files'],1)
                output=base/'raw'/record['files'][0]['path'];output.chmod(0o600);output.write_bytes(b'X'*len(data))
                with self.assertRaises(ValueError):m.target_finalize(base,{})

    def test_strict_routes_no_private_key_copy_secret_argv_or_proxy(self):
        for argv in (m.ssh_k3(),m.ssh_target()):
            self.assertIn('-oStrictHostKeyChecking=yes',argv)
            self.assertIn('-oProxyCommand=none',argv)
            self.assertNotIn('-oStrictHostKeyChecking=no',argv)
            self.assertNotIn('-i',argv)
        self.assertEqual(m.clean_env('RAM_FIXTURE_PASSWORD')['SSHPASS'],'RAM_FIXTURE_PASSWORD')
        with patch.dict('os.environ',{'SSHPASS':'old','HF_TOKEN':'old'}):
            self.assertNotIn('SSHPASS',m.clean_env());self.assertNotIn('HF_TOKEN',m.clean_env())
        # actual command generation, no network or payload data.
        calls=[]
        with tempfile.TemporaryDirectory() as temp,patch.object(m,'remote',side_effect=lambda ssh,cmd,*a,**k:calls.append(cmd) or {}):
            result=m.install_code(m.ssh_k3(),Path(temp).resolve()/'snapshot')
        self.assertTrue(str(result).endswith('peer_newseed_receive.py'))
        self.assertTrue(all(len(cmd.encode())<16000 for cmd in calls))
        self.assertFalse(any('RAM_FIXTURE_PASSWORD' in cmd for cmd in calls))

    def test_rsync_transport_excludes_target_host_and_secret(self):
        commands=[]
        class Process:
            returncode=0
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def poll(self):return 0
        with tempfile.TemporaryDirectory() as temp,patch.object(m.subprocess,'Popen',side_effect=lambda args,**kw:commands.append((args,kw)) or Process()):
            source=Path(temp).resolve()/'own.tar';source.write_bytes(b'x')
            m.transfer(m.ssh_target(),source,Path('/approved/own.tar.part'),Path('/approved/own.lock'),source.parent,'TEST',m.clean_env('RAM_FIXTURE_PASSWORD'))
        argv,kwargs=commands[0];transport=argv[argv.index('-e')+1]
        self.assertNotIn('mou176@',transport);self.assertTrue(argv[-1].startswith('mou176@10.120.81.30:'))
        self.assertNotIn('RAM_FIXTURE_PASSWORD',str(argv));self.assertEqual(kwargs['env']['SSHPASS'],'RAM_FIXTURE_PASSWORD')
        self.assertIn('flock -n',str(argv))

    def test_download_exact_range_resume_and_sha(self):
        c,data=fixture();record=c['groups'][0];record['archive_bytes']=len(data);record['archive_sha256']=hashlib.sha256(data).hexdigest()
        class Response:
            status_code=206;headers={'Content-Range':'bytes 4-%d/%d'%(len(data)-1,len(data))}
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def raise_for_status(self):pass
            def iter_content(self,n):yield data[4:]
        class Session:
            def get(self,url,**kwargs):
                self.kwargs=kwargs;return Response()
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp).resolve();m.secure_dir(base/'archives');final,partial=m.paths(base,record);partial.write_bytes(data[:4]);session=Session()
            self.assertEqual(m.download(session,base,record,'a'*40),final)
            self.assertEqual(final.read_bytes(),data);self.assertEqual(session.kwargs['headers'],{'Range':'bytes=4-'})
            self.assertFalse(partial.exists())

    def test_ignored_range_preserves_partial_and_never_accepts(self):
        c,data=fixture();record=c['groups'][0];record['archive_bytes']=len(data)
        class Response:
            status_code=200;headers={}
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def raise_for_status(self):pass
        class Session:
            def get(self,*args,**kwargs):return Response()
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp).resolve();m.secure_dir(base/'archives');final,partial=m.paths(base,record);partial.write_bytes(data[:4])
            with self.assertRaises(ValueError):m.download(Session(),base,record,'a'*40)
            self.assertEqual(partial.read_bytes(),data[:4]);self.assertFalse(final.exists())


if __name__=='__main__':unittest.main()
