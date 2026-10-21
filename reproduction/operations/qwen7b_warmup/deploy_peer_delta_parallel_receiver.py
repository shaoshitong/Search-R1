"""New receiver transport only; never changes an existing receiver launch."""
import sys,shlex,importlib.util
from . import deploy_peer_frozen_delta_transfer as t
from . import peer_newseed_receive_parallel as receiver

def receive_action(repo,commit,args):
 raw=t.blob(repo,commit,t.P+'peer_newseed_receive_parallel.py')
 if raw!=t.Path(receiver.__file__).read_bytes():raise ValueError('Parallel receiver differs from published immutable bytes')
 base=receiver.K2/t.SNAPSHOT;digest=t.sha(raw);code=base/'code'/digest/'peer_newseed_receive.py';v=dict(base=str(base),snapshot=t.SNAPSHOT,code=str(code),code_sha256=digest,source_commit=commit,prefix=receiver.PREFIX+t.SNAPSHOT,catalog_commit=args.catalog_commit,hf_proxy=args.hf_proxy)
 with t.recv.route(False,False,args.via_hkust1)as ssh:
  if args.action=='receive-stage':
   actual=receiver.install_code(ssh,base);assert actual==code;return dict(status='PUBLISHED_ISOLATED_PARALLEL_RANGE_RECEIVER_STAGED',code=str(code),sha256=digest,source_commit=commit,workers=8)
  secret=None
  if args.action=='receive-start':
   if not args.catalog_commit:raise ValueError('Actual complete catalog required')
   path=args.control_root/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py';spec=importlib.util.spec_from_file_location('parallel_delta_private',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);secret=dict(hf_token=t.recv.token(),hpc3_password=m.password_for('mou176').decode())
  return receiver.remote(ssh,'python3 -c '+shlex.quote(t.recv.render(v,args.action[8:])),secret,timeout=55)

def main():
 if '--action'not in sys.argv or sys.argv[sys.argv.index('--action')+1]not in('receive-stage','receive-start','receive-status'):raise ValueError('Only explicit new receiver stage/start/status; no other phase mutation')
 t.receive_action=receive_action;t.main()
if __name__=='__main__':main()
