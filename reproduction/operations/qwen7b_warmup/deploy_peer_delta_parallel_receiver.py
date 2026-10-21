"""New receiver transport only; never changes an existing receiver launch."""
import sys,shlex,importlib.util
from . import deploy_peer_frozen_delta_transfer as t
from . import peer_newseed_receive_parallel as receiver

def receive_action(repo,commit,args):
 raw=t.blob(repo,commit,t.P+'peer_newseed_receive_parallel.py')
 if raw!=t.Path(receiver.__file__).read_bytes():raise ValueError('Parallel receiver differs from published immutable bytes')
 base=receiver.K2/t.SNAPSHOT;digest=t.sha(raw);code=base/'code'/digest/'peer_newseed_receive.py';v=dict(resume_failed=args.action=='receive-resume-failed',base=str(base),snapshot=t.SNAPSHOT,code=str(code),code_sha256=digest,source_commit=commit,prefix=receiver.PREFIX+t.SNAPSHOT,catalog_commit=args.catalog_commit,hf_proxy=args.hf_proxy)
 with t.recv.route(False,False,args.via_hkust1)as ssh:
  if args.action=='receive-stage':
   actual=receiver.install_code(ssh,base);assert actual==code;return dict(status='PUBLISHED_ISOLATED_PARALLEL_RANGE_RECEIVER_STAGED',code=str(code),sha256=digest,source_commit=commit,workers=8)
  secret=None
  if args.action in('receive-start','receive-resume-failed'):
   if not args.catalog_commit:raise ValueError('Actual complete catalog required')
   path=args.control_root/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py';spec=importlib.util.spec_from_file_location('parallel_delta_private',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);secret=dict(hf_token=t.recv.token(),hpc3_password=m.password_for('mou176').decode())
  if args.action in('receive-start','receive-resume-failed'):
   prelude=r"""
 if inputs.get('resume_failed')and(b/'launch.json').exists():
  previous=json.loads((b/'launch.json').read_text())
  if not live(previous)and(b/'failure.json').exists():
   failed=json.loads((b/'failure.json').read_text());assert failed.get('error_type')in('SSLError','ConnectionError','ConnectTimeout','ReadTimeout')and failed['epoch']>=previous['epoch'],'Only known failed transport retry authorized'
   assert not(b/'controller.json').exists()and not list((b/'target-accepted').glob('*.json'))and not list((b/'download-accepted').glob('*.json')),'This bounded retry is metadata-before-payload only'
   receiver_guard=(b/'receiver.lock').open('a');fcntl.flock(receiver_guard,fcntl.LOCK_EX|fcntl.LOCK_NB);receiver_guard.close()
   attempts=b/'failed-attempts';attempts.mkdir(exist_ok=True);number=len(list(attempts.glob('attempt-*')))+1;assert number<=3,'Bounded owned retry limit reached';history=attempts/('attempt-%03d'%number);history.mkdir();proofs={}
   for name in('launch.json','launch-intent.json','failure.json','retry.json','progress.json'):
    q=b/name
    if q.exists():
     assert not q.is_symlink();raw=q.read_bytes();dest=history/name;q.rename(dest);proofs[name]=dict(path=str(dest),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
   atomic(history/'preserved.json',dict(status='FAILED_METADATA_TLS_ATTEMPT_PRESERVED_BEFORE_OWN_RETRY',files=proofs,payload_files_changed=False,previous_process_dead=True,new_proxy=inputs['hf_proxy'],epoch=time.time()))
"""
   start=t.recv.START.replace(" if (b/'launch.json').exists():",prelude+" if (b/'launch.json').exists():",1)
   script='inputs='+repr(v)+'\n'+t.recv.COMMON+start
  else:script=t.recv.render(v,'status')
  return receiver.remote(ssh,'python3 -c '+shlex.quote(script),secret,timeout=55)

def main():
 if '--action'not in sys.argv or sys.argv[sys.argv.index('--action')+1]not in('receive-stage','receive-start','receive-status','receive-resume-failed'):raise ValueError('Only explicit new receiver stage/start/status; no other phase mutation')
 t.ACTIONS=t.ACTIONS+('receive-resume-failed',);t.receive_action=receive_action;t.main()
if __name__=='__main__':main()
