"""Small remote CPU-only deployment of existing prediction paired analysis."""
import argparse,base64,hashlib,importlib.util,json,shlex,subprocess,zlib
from pathlib import Path
R=Path(__file__).resolve().parents[3];CONTROL=R.parents[1];P='reproduction/operations/qwen7b_warmup/';BRANCH='codex/mixed-direct-paired-stability-20261022';A='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021';BASE=A+'/mixed-direct-E0-CP128-CP512-paired-stability-v1';OUT=R/'reproduction/reports/qwen7b-algorithm-improvement-20261021/mixed-direct-paired-stability';WORKER=P+'paired_mixed_direct_snapshots.py';PROTOCOL=P+'paired_question_stability.py';PROTOCOL_ORIGIN='1a0f5b411e9c31ff9a3b64a02aa9644156eee6c0'
def git(c,n):return subprocess.check_output(['git','show',c+':'+n],cwd=R)
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--action',choices=('launch','status','result'),required=True);args=parser.parse_args();ref='refs/heads/'+BRANCH;commit=subprocess.check_output(['git','rev-parse',ref],cwd=R).decode().strip();assert subprocess.check_output(['git','ls-remote','origin',ref],cwd=R).decode().split()[0]==commit
 source=git(commit,WORKER);assert source==(R/WORKER).read_bytes();protocol=git(PROTOCOL_ORIGIN,PROTOCOL);files={WORKER:source,PROTOCOL:protocol};codehash=hashlib.sha256(source+protocol).hexdigest();prior=R/'reproduction/reports/qwen7b-algorithm-improvement-20261021/mixed-direct-cp512-eval';direct=json.loads((prior/'actual-readonly-audit.json').read_text());evidence=json.loads((prior/'actual-readonly-evidence.json').read_text());pred={}
 for phase,kind,key in [('E0','raw','E0'),('E0','EMA','E0_EMA'),('CP128','raw','CP128_raw'),('CP128','EMA','CP128_EMA'),('CP512','raw','CP512_raw'),('CP512','EMA','CP512_EMA')]:pred[key]=direct['prediction_refs'][phase][kind]['fixed_selection2048']
 e2=json.loads((CONTROL/'outputs/qwen7b-warmup-20261019/reports/E2-same-HF-independent-final-audit.json').read_text());e4=json.loads((CONTROL/'outputs/qwen7b-warmup-20261019/reports/peer8192-group-head-input-refs.json').read_text());pred['E2']=e2['predictions'];pred['E4']=e4['E4_predictions'];manifest=dict(role='already_viewed_recipe_selection_validation',questions=2048,bootstrap_replicates=2000,predictions=pred,accepted_metadata=dict(direct_identity_audit=evidence['independent_audit'],E2_complete=e2['complete'],E4_complete=e4['E4_baseline_child'],same_HF_reader=e4['validation_reader_audit']),protocol_module=dict(origin=PROTOCOL_ORIGIN,path=PROTOCOL,sha256=hashlib.sha256(protocol).hexdigest()),adapter_source=dict(commit=commit,path=WORKER,sha256=hashlib.sha256(source).hexdigest()),score_semantics=dict(direct='learned_direct_realG5_Mixed_probability',old_E2_E4='stable_independent_Bernoulli_event_probability'),threshold=.5,seed=9137,no_new_scoring=True,final4096_read=False)
 v=dict(base=BASE,files={n:base64.b64encode(raw).decode()for n,raw in files.items()},hashes={n:hashlib.sha256(raw).hexdigest()for n,raw in files.items()},codehash=codehash,worker=WORKER,manifest=manifest,action=args.action)
 body='''import pathlib,json,hashlib,base64,os,subprocess,time,fcntl
b=pathlib.Path(v['base']);b.mkdir(parents=True,exist_ok=True);code=b/('code-'+v['codehash'][:12]);output=b/'paired-stability.json'
def proof(p):
 raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def read(p):return json.loads(p.read_bytes())
def once(p,raw):
 if p.exists():assert p.read_bytes()==raw
 else:
  with p.open('xb')as f:f.write(raw)
if v['action']=='launch':
 guard=(b/'launch-control.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if output.exists():print(json.dumps(dict(status='EXISTING_COMPLETED_PAIRED_RESULT',output=proof(output))));raise SystemExit(0)
 assert not(b/'launch-intent.json').exists(),'Prior launch intent requires actual PID/output reconciliation'
 for name,encoded in v['files'].items():
  raw=base64.b64decode(encoded);assert hashlib.sha256(raw).hexdigest()==v['hashes'][name];p=code/name;p.parent.mkdir(parents=True,exist_ok=True);once(p,raw);compile(raw,str(p),'exec');p.chmod(0o444)
 config=(json.dumps(v['manifest'],sort_keys=True,separators=(',',':'))+chr(10)).encode();once(b/'manifest.json',config);once(b/'launch-intent.json',json.dumps(dict(manifest=proof(b/'manifest.json'),source_sha256=v['hashes'],epoch=time.time()),sort_keys=True).encode());log=(b/'cpu.log').open('ab');env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='');child=subprocess.Popen(['python3','-u',str(code/v['worker']),'--manifest',str(b/'manifest.json'),'--output',str(output)],stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True);result=dict(status='BOUNDED_REMOTE_CPU_PAIRED_ANALYSIS_STARTED',pid=child.pid,manifest=proof(b/'manifest.json'),code_sha256=v['hashes'],GPU_requested=False);once(b/'launch.json',(json.dumps(result,sort_keys=True)+chr(10)).encode());print(json.dumps(result))
elif v['action']=='status':
 result=dict(epoch=time.time(),complete=output.exists())
 if(b/'launch.json').exists():
  result['launch']=read(b/'launch.json');pid=result['launch']['pid'];p=pathlib.Path('/proc')/str(pid)
  if p.exists():result['process_stat']=(p/'stat').read_text().rsplit(')',1)[1].split()[:22]
 if(b/'cpu.log').exists():result['log_tail']=(b/'cpu.log').read_text()[-2000:]
 if output.exists():result['output']=proof(output)
 print(json.dumps(result))
else:
 x=read(output);assert x['status']=='DESCRIPTIVE_DIRECT_E0_CP128_CP512_AND_E2_E4_PAIRED_QUESTION_STABILITY_COMPLETE';summary={k:val for k,val in x.items()if k!='strata'};summary['strata']={k:val for k,val in x['strata'].items()if k!='strata'};summary['remote_complete']=proof(output);print(json.dumps(summary))
'''
 text='v='+repr(v)+'\n'+body;compile(text,'paired_remote_control','exec');packed=base64.b64encode(zlib.compress(text.encode())).decode();command='python3 -c '+shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(packed)+')))');assert len(command.encode())<16000
 spec=importlib.util.spec_from_file_location('private_paired_direct',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl);ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0']).decode().strip();r=ctl.run(command,account='mou176',source_address=ip,timeout=55)
 if r.returncode:raise RuntimeError(r.stdout[-1000:]+r.stderr[-2000:])
 result=json.loads(r.stdout.splitlines()[-1]);OUT.mkdir(parents=True,exist_ok=True);(OUT/('actual-'+args.action+'.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
