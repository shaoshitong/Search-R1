"""Small source/interface/scope tests; no model, remote calls or GPU submit."""
import ast,base64,contextlib,gzip,hashlib,io,json,pathlib,shlex,subprocess,tarfile,tempfile,unittest,zlib
from unittest import mock
from . import deploy_joint_balanced_native_full169028 as d

class ScopeTests(unittest.TestCase):
 def test_missing_and_wrong_bank_fail_before_config(self):
  import fcntl,os,time
  for wrong in (False,True):
   with tempfile.TemporaryDirectory()as tmp:
    b=pathlib.Path(tmp)/'run';b.mkdir();bank=pathlib.Path(tmp)/'bank';bank.mkdir()
    if wrong:(bank/'complete.json').write_text(json.dumps(dict(status='FROZEN_FIVE_BANK_REAL_G10_169028_NATIVE_JOINT_INPUTS_ACCEPTED',questions=124194)))
    env=dict(b=b,v=dict(bank=str(bank)),pathlib=pathlib,fcntl=fcntl,os=os,json=json,time=time,active=lambda:[],read=lambda p:json.loads(pathlib.Path(p).read_text()),proof=lambda p:dict(path=str(p),bytes=pathlib.Path(p).stat().st_size,sha256=hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()))
    with contextlib.redirect_stdout(io.StringIO())as capture:
     if wrong:
      with self.assertRaises(AssertionError):exec(d.PREPARE,env)
     else:
      with self.assertRaises(SystemExit)as e:exec(d.PREPARE,env)
      self.assertEqual(e.exception.code,0);self.assertIn('WAITING_FOR_ACTUAL_FULL169028',capture.getvalue())
    self.assertFalse((b/'config.json').exists());self.assertFalse((b/'training.sbatch').exists())
 def test_actual_config_initializer_matches_published_worker_protocol(self):
  ns={};exec(subprocess.check_output(['git','show',d.ROOT+':'+d.PKG+'full169028_protocol.py'],cwd=d.R),ns)
  node=next(n for n in ast.parse(d.PREPARE).body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='c'for t in n.targets))
  ref=dict(path='/remote/metadata',bytes=1,sha256='a'*64);old={k:ref for k in('source_receipt','source_config','source_bank','model_config')};old['model_dir']='/remote/model'
  env=dict(b=pathlib.Path(d.BASE),code=pathlib.Path('/remote/immutable-code'),v=dict(pkg=d.PKG,sampler=d.SAMPLER,runtime_commit=d.ROOT,base_runtime_commit=d.BASE_RUNTIME,worker_origin=d.WORKER_ORIGIN),complete_ref=ref,gr=ref,fr=ref,exclusion='/remote/exclusions',proof=lambda p:ref,source=dict(sha256='e29168c36de152a57294f055e7d420cbba25cae31aeace32e793ba1098bf97a8',steps=14418),old=old,pub=dict(files={d.PKG+'balanced_hard_g10_sampler.py':dict(sha256='b'*64)}),dfeatures=ref,r2ref=ref,valcomplete=ref,audit=ref,groups=[1],hashes={'x'},baseline_publication_ref=ref)
  exec(compile(ast.Module(body=[node],type_ignores=[]),'actual-config-initializer','exec'),env);c=env['c'];ns['validate_config'](d.BASE,c)
  self.assertEqual(c['batch_questions'],32);self.assertEqual(c['steps_per_epoch']*c['epochs'],21132);self.assertEqual(c['checkpoint_policy']['immutable_pins'],[k for k in range(21133)if ns['pinned'](k)])
 def test_published_mixed_origin_bundle_and_manifest(self):
  raw=d.bundle('controller-fixture');contents={}
  with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz')as archive:
   for row in archive:contents[row.name]=archive.extractfile(row).read()
  self.assertEqual(len(contents),23)
  for name,value in contents.items():
   origin=d.SAMPLER if name==d.PKG+'balanced_hard_g10_sampler.py'else d.ROOT if name in d.ROOT_FILES else d.BASE_RUNTIME
   self.assertEqual(value,subprocess.check_output(['git','show',origin+':'+name],cwd=d.R))
  with tempfile.TemporaryDirectory()as tmp:
   b=pathlib.Path(tmp);code=b/'code';code.mkdir();(code/'upload-0000.part').write_bytes(raw)
   def immutable(p,v):p.write_text(json.dumps(v))
   def proof(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
   v=dict(parts=1,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),commit='controller-fixture',runtime_commit=d.ROOT,base_runtime_commit=d.BASE_RUNTIME,sampler=d.SAMPLER,root_files=d.ROOT_FILES,runtime_branch=d.ROOT_BRANCH,worker_origin=d.WORKER_ORIGIN,controller=dict(path='controller',sha256='c'*64))
   with contextlib.redirect_stdout(io.StringIO()):exec(d.EXTRACT,dict(b=b,code=code,v=v,tarfile=tarfile,io=io,pathlib=pathlib,hashlib=hashlib,json=json,immutable=immutable,proof=proof))
   pub=json.loads((b/'source-publication.json').read_text());self.assertEqual(pub['qualified_runtime_base_commit'],d.BASE_RUNTIME)
   self.assertEqual(pub['file_origins'][d.PKG+'native_joint_rolling_checkpoint.py'],d.BASE_RUNTIME);self.assertEqual(pub['file_origins'][d.PKG+'full169028_protocol.py'],d.ROOT)
 def test_resource_entry_and_bounded_control(self):
  entry=d.ENTRY.replace('@@BASE@@',d.BASE).replace('@@CODE@@','/remote/immutable-code');self.assertIn('#SBATCH --time=2-00:00:00',entry);self.assertIn('#SBATCH --qos=normal',entry);self.assertNotIn('--dependency',entry);self.assertIn('joint_balanced_native_full169028',entry)
  with tempfile.NamedTemporaryFile('w')as f:f.write(entry);f.flush();subprocess.run(['bash','-n',f.name],check=True)
  v=dict(base='/remote/new',A=d.A,bank=d.BANK,pkg=d.PKG,commit='e'*40,sampler=d.SAMPLER,runtime_commit=d.ROOT,runtime_branch=d.ROOT_BRANCH,base_runtime_commit=d.BASE_RUNTIME,root_files=d.ROOT_FILES,worker_origin=d.WORKER_ORIGIN,entry=d.ENTRY)
  script='v='+repr(v)+'\n'+d.COMMON+d.PREPARE;compile(script,'actual-controller','exec');packed=base64.b64encode(zlib.compress(script.encode())).decode();cmd='python3 -c '+shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(packed)+')))');self.assertLess(len(cmd.encode()),16000)
 def test_submit_requires_separate_explicit_authorization(self):
  with mock.patch('sys.argv',['deployer','--action','submit']),mock.patch.object(d.subprocess,'check_output',side_effect=AssertionError('network must not be reached')),contextlib.redirect_stderr(io.StringIO()):
   with self.assertRaises(SystemExit)as e:d.main()
  self.assertEqual(e.exception.code,2)
if __name__=='__main__':unittest.main()
