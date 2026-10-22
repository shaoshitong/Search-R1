import pathlib,json,subprocess,os
base=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-natural-g5-8192-20261022-v1')
pub=json.loads((base/'source-publication.json').read_text());runtime=pathlib.Path(pub['code_root'])/'reproduction/operations/qwen7b_warmup/runtime.sh'
script=base/'postrun-audit-code-c91222bd3215/compare_natural_g5_online_matched.py'
env=dict(os.environ,QWEN7B_CODE_ROOT=pub['code_root'],CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
r=subprocess.run(['bash',str(runtime),'-u',str(script),'--layers','21','27','--epochs','0','1','2','3'],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=40)
if r.returncode:raise RuntimeError(r.stderr[-3000:]+r.stdout[-3000:])
print(r.stdout,end='')
