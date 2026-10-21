"""Stage/start/status one typed CPU milestone audit; never submits GPU work."""
import argparse,base64,hashlib,importlib.util,json,pathlib,shlex,subprocess
R=pathlib.Path(__file__).resolve().parents[4];C=R.parents[1];HERE=pathlib.Path(__file__).parent
NAME='reproduction/operations/qwen7b_warmup/audit_direct_count_headlr.py'
BASE='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/direct-hidden-count30016-headlr1e4-v1'
COMMON='''import pathlib,json,hashlib,fcntl,os,time,subprocess,base64,sys
b=pathlib.Path(v['base']);root=b/'typed-milestone-auditor';root.mkdir(exist_ok=True);guard=(root/'operator.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
def read(p):return json.loads(pathlib.Path(p).read_bytes())
def proof(p):
 p=pathlib.Path(p);raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
'''
def main():
 p=argparse.ArgumentParser();p.add_argument('--action',choices=('stage','start','status'),required=True);p.add_argument('--cursor',type=int,choices=(256,512,768,938),default=256);a=p.parse_args()
 s=importlib.util.spec_from_file_location('typed_headlr_audit_private',C/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(s);s.loader.exec_module(ctl);ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0']).decode().strip()
 def call(body,**values):
  values.update(base=BASE,cursor=a.cursor);cmd='python3 -c '+shlex.quote('v='+repr(values)+'\n'+COMMON+body)
  if len(cmd.encode())>=16000:raise ValueError('Under16KB control commands required')
  r=ctl.run(cmd,account='mou176',source_address=ip,timeout=55)
  if r.returncode:raise RuntimeError('Typed CPU audit not accepted; inspect preserved outcome, no blind retries: '+r.stderr.decode()[-1800:]+r.stdout.decode()[-1800:])
  return json.loads(next(x for x in reversed(r.stdout.decode().splitlines())if x.startswith('{')))
 if a.action=='stage':
  commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R).decode().strip();remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/qwen7b-probe-warmup-20261019'],cwd=R).decode().split()[0]
  if commit!=remote:raise ValueError('Publish typed auditor before staging')
  raw=subprocess.check_output(['git','show',commit+':'+NAME],cwd=R)
  for i,start in enumerate(range(0,len(raw),4096)):
   part=raw[start:start+4096];call("code=root/('code-'+v['commit'][:12]);code.mkdir(exist_ok=True);p=code/('part-%04d'%v['i']);raw=base64.b64decode(v['data']);assert hashlib.sha256(raw).hexdigest()==v['sha']\nif p.exists():assert p.read_bytes()==raw\nelse:p.write_bytes(raw)\nprint(json.dumps(dict(part=v['i'])))",commit=commit,i=i,data=base64.b64encode(part).decode(),sha=hashlib.sha256(part).hexdigest())
  result=call("code=root/('code-'+v['commit'][:12]);raw=b''.join((code/('part-%04d'%i)).read_bytes()for i in range(v['parts']));assert hashlib.sha256(raw).hexdigest()==v['sha'];source=code/'audit.py';compile(raw,str(source),'exec')\nif source.exists():assert source.read_bytes()==raw\nelse:source.write_bytes(raw);source.chmod(0o444)\npub=dict(commit=v['commit'],remote_sha=v['commit'],source=proof(source));p=root/'source-publication.json'\nif p.exists():assert read(p)==pub\nelse:p.write_text(json.dumps(pub)+'\\n')\nprint(json.dumps(dict(publication=proof(p))))",commit=commit,parts=(len(raw)+4095)//4096,sha=hashlib.sha256(raw).hexdigest())
 elif a.action=='start':
  result=call("""folder=b/'evaluations'/('cursor-%06d'%v['cursor']);assert (folder/'complete.json').exists(),'Wait for real fixed evaluation complete'
if (folder/'independent-audit.json').exists():print(json.dumps(dict(status='EXISTING_TYPED_AUDIT_PRESERVED',receipt=proof(folder/'independent-audit.json'))))
else:
 work=root/('cursor-%06d-operator-v2'%v['cursor']);work.mkdir(exist_ok=True)
 legacy=root/('cursor-%06d'%v['cursor'])
 if (legacy/'controller.json').exists():
  old=read(legacy/'controller.json');text=(legacy/'cpu.log').read_text()
  assert 'SyntaxError: EOL while scanning string literal'in text and not pathlib.Path('/proc/'+str(old['pid'])).exists(),'Only the known exited pre-execution wrapper syntax failure may be recovered'
  (work/'preserved-attempt01.json').write_text(json.dumps(dict(controller=proof(legacy/'controller.json'),log=proof(legacy/'cpu.log'),failure='WRAPPER_PARSE_BEFORE_AUDIT_EXECUTION',no_model_or_training_state_modified=True))+chr(10))
 if (work/'controller.json').exists():print(json.dumps(dict(status='EXISTING_TYPED_CPU_AUDIT_CONTROLLER_PRESERVED',controller=read(work/'controller.json'))));sys.exit(0)
 pub=read(root/'source-publication.json');assert proof(pub['source']['path'])==pub['source'];runtime=read(b/'source-publication.json')['code_root']
 code="import importlib.util,json; s=importlib.util.spec_from_file_location('reproduction.operations.qwen7b_warmup.typed_headlr_actual_auditor',"+repr(pub['source']['path'])+"); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); r=m.audit("+repr(str(b))+","+str(v['cursor'])+",checkpoint_metadata_only="+str(v['cursor']==938)+"); print(json.dumps(dict(status=r['status'],cursor=r['cursor'])))"
 args=['bash',runtime+'/reproduction/operations/qwen7b_warmup/runtime.sh','-u','-c',code]
 if v['cursor']!=938:args=['srun','--jobid=743809','--overlap','--gres=gpu:0','--ntasks=1','--cpus-per-task=2']+args
 env=dict(os.environ,QWEN7B_CODE_ROOT=runtime,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
 wrapper="import subprocess,pathlib,json,time; r=subprocess.run("+repr(args)+"); pathlib.Path("+repr(str(work/'exit.json'))+").write_text(json.dumps(dict(returncode=r.returncode,epoch=time.time()))+chr(10)); raise SystemExit(r.returncode)"
 compile(wrapper,str(work/'wrapper.py'),'exec');(work/'wrapper.py').write_text(wrapper+chr(10))
 with (work/'cpu.log').open('ab',0)as log:p=subprocess.Popen(['python3','-u',str(work/'wrapper.py')],stdout=log,stderr=subprocess.STDOUT,env=env,stdin=subprocess.DEVNULL,start_new_session=True)
 result=dict(status='ONE_TYPED_CPU_MILESTONE_AUDIT_STARTED',pid=p.pid,cursor=v['cursor'],config_sha256='71c3ee90f0232b889d4ced2d0cb50d69e852eb1ebbdbb55fdff03eaefdbbcc71',head_LR=1e-4,existing_allocation_CPU_step=v['cursor']!=938,terminal_no_tensor_load=v['cursor']==938,source=pub['source'],epoch=time.time());(work/'controller.json').write_text(json.dumps(result)+'\\n');print(json.dumps(result))""")
 else:
  result=call("work=root/('cursor-%06d-operator-v2'%v['cursor']);p=b/'evaluations'/('cursor-%06d'%v['cursor'])/'independent-audit.json';out=dict(cursor=v['cursor'],epoch=time.time())\nif (work/'controller.json').exists():out['controller']=read(work/'controller.json')\nif (work/'exit.json').exists():out['exit']=read(work/'exit.json')\nif p.exists():out.update(receipt=proof(p),result=read(p))\nif (work/'cpu.log').exists()and not p.exists():out['log_tail']=(work/'cpu.log').read_text()[-2000:]\nprint(json.dumps(out))")
 (HERE/('actual-milestone-%06d-%s.json'%(a.cursor,a.action))).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
