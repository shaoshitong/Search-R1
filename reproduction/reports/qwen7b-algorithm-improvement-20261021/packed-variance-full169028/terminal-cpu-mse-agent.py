"""Route the published terminal-only CPU acceptance through one selected control.

Reuses commit799e75f3's complete remote staging/start/status implementation;
no Slurm mutation, GPU model load, or final-holdout reads.
"""
import argparse,base64,hashlib,json,pathlib,subprocess,sys,tempfile
B=pathlib.Path(__file__).resolve().parent;REPO=B.parents[3]
P='reproduction/operations/qwen7b_warmup/'
BRANCH='codex/q7b-full169028-packed-cpu-acceptance-20261022'
def main():
 p=argparse.ArgumentParser();p.add_argument('--action',choices=('stage','start','status'),required=True);p.add_argument('--commit',required=True);p.add_argument('--control',choices=('balanced','campus'),required=True);a=p.parse_args()
 actual=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+BRANCH],cwd=REPO,text=True).split()[0];subprocess.check_call(['git','merge-base','--is-ancestor',a.commit,actual],cwd=REPO)
 deploy=subprocess.check_output(['git','show',a.commit+':'+P+'deploy_full169028_packed_CPU_acceptance.py'],cwd=REPO)
 namespace={'__name__':'published_acceptance_implementation','__file__':str(REPO/P/'deploy_full169028_packed_CPU_acceptance.py')};exec(compile(deploy,'published_terminal_acceptance','exec'),namespace)
 raw=subprocess.check_output(['git','show',a.commit+':'+P+'accept_full169028_packed_variance.py'],cwd=REPO)
 v=dict(base=namespace['BASE'],commit=a.commit,source=base64.b64encode(raw).decode(),sha256=hashlib.sha256(raw).hexdigest())
 # Share the actual8GB-login CPU mutex with the new matched acceptor.
 # Pass the locked open file description into the child runtime so the mutex
 # persists until that one validator exits; never starts another CPU torchload.
 shared='''shared_path=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-mixed-P80-handoff-CPU-acceptance-20261023-v1/acceptance.lock')
shared_path.parent.mkdir(exist_ok=True);shared_guard=shared_path.open('a+')
try:fcntl.flock(shared_guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:
 print(json.dumps(dict(status='WAITING_SHARED_CPU_MUTEX_NOT_STARTED',shared_CPU_lock=str(shared_path),CPU_validator_started=False)));raise SystemExit(0)
'''
 start=shared+namespace['START'].replace('start_new_session=True)','start_new_session=True,pass_fds=(shared_guard.fileno(),))')
 body=namespace['STATUS']if a.action=='status'else namespace['STAGE']+(start if a.action=='start'else'print(json.dumps(stage()[3]))\n')
 script='v='+repr(v)+'\n'+namespace['COMMON']+body;compile(script,'terminal_CPU_selected_route','exec')
 control=REPO.parents[1]/('outputs/qwen7b-warmup-20261019/'+a.control+'-control.py')
 with tempfile.TemporaryDirectory(prefix='mse-terminal-CPU-code-')as tmp:
  path=pathlib.Path(tmp)/'terminal-cpu.py';path.write_text(script)
  r=subprocess.run([sys.executable,str(control),str(path),'--timeout','35'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
 if r.returncode:raise RuntimeError((r.stdout+r.stderr)[-2400:])
 result=json.loads(r.stdout.splitlines()[-1]);result['explicit_control_route']=a.control;result['published_implementation_commit']=a.commit;result['published_deployer_sha256']=hashlib.sha256(deploy).hexdigest();print(json.dumps(result))
if __name__=='__main__':main()
