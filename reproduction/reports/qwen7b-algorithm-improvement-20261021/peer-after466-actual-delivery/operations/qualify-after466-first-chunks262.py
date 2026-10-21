import pathlib,subprocess,os,json,hashlib,time
A=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021')
n='262';assert n in('243','262');b=A/('peer-joint124194-audited'+n+'-L26-v1');code=b/'cpu-bank-acceptance-c1d260806090';p=b/'first-four-chunks-independent-audit.json'
def ref(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
if p.exists():v=json.loads(p.read_text());print(json.dumps(dict(status='EXISTING_FOUR_FIRST_CHUNK_ACCEPTANCE_PRESERVED',audit=ref(p),summary=v)))
else:
 assert all((b/('shard-%02d'%i)/'capture/chunk-00000/complete.json').exists()for i in range(4))
 env=dict(os.environ,QWEN7B_CODE_ROOT=str(code),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
 args=['bash',str(code/'reproduction/operations/qwen7b_warmup/runtime.sh'),'-c','from pathlib import Path; from reproduction.operations.qwen7b_warmup.peer_joint_subset_l26_merge import qualify_first; qualify_first(Path('+repr(str(b))+'))']
 allowed=sorted(os.sched_getaffinity(0))[:2];t=time.time();r=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=env,universal_newlines=True,timeout=45,preexec_fn=lambda:os.sched_setaffinity(0,allowed));(b/'first-chunk-independent-control.log').write_text(r.stdout);assert r.returncode==0,r.stdout[-1800:];v=json.loads(p.read_text());print(json.dumps(dict(status='FOUR_ACTUAL_FIRST_CHUNKS_FRESH_SHA_AND_READER_EXITED_ZERO',audit=ref(p),summary=v,elapsed=time.time()-t)))
