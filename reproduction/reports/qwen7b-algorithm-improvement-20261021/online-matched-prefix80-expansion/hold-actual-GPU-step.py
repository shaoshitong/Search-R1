import pathlib,json,subprocess,time
S=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-mixed-layer-prefix-20261022-2gpu-v1');out={}
def run(c):
 p=subprocess.run(c,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=10);return {'rc':p.returncode,'stdout':p.stdout.strip(),'stderr':p.stderr.strip()}
for layer in(15,18):
 b=S/('layer%d-prefix80'%layer);r=json.loads((b/'root-submission.json').read_bytes());job=r['job_id'];cp=json.loads((b/'latest.json').read_bytes());progress=[json.loads((b/('progress-rank-%d.json'%k)).read_bytes())for k in(0,1)];o={'time':time.time(),'job_id':job,'checkpoint':cp,'cursors':[p['cursor']for p in progress]}
 if cp['path'].endswith('epoch-02.pt')and all(p['cursor']==128 for p in progress):
  o['steps']=run(['squeue','--steps','-h','-j',job,'-o','%i|%j|%T']);o['signal']=run(['scancel','--signal=STOP',job+'.0']);o['eval2_exists']=(b/'evaluation-epoch-02.json').exists();o['note']='GPU-step STOP requested; actual rank T must be verified before cancellation; no rollback authorized.'
  (b/'GPU-step-stop-request.json').write_text(json.dumps(o,sort_keys=True)+'\n')
 out[str(layer)]=o
print(json.dumps(out))
