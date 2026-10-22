"""Sequential CPU acceptance of the two already-held complete epoch files."""
import pathlib,json,subprocess,hashlib,time
A=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021');S=A/'online-mixed-layer-prefix-20261022-2gpu-v1';stage=json.loads((A/'online-mixed-P80-handoff-CPU-acceptance-20261023-v1/staged.json').read_bytes())
assert stage['validator']['sha256']=='01aa43ccc117370ef8595d953e8e41f99ad3d52e9302edcdc18cd76fc31a174d'
entry=pathlib.Path(stage['CPU_entry']['path']);assert hashlib.sha256(entry.read_bytes()).hexdigest()==stage['CPU_entry']['sha256'];out={'time':time.time(),'results':{}}
for layer in(18,15):
 b=S/('layer%d-prefix80'%layer);held=json.loads((b/'walltime-boundary-held.json').read_bytes());assert held['status']=='COMPLETE_EPOCH_BOUNDARY_HELD_FOR_CPU_ACCEPTANCE'
 p=subprocess.run(['bash',str(entry),'--layer',str(layer),'--prefix','80','--expected-epoch',str(held['epoch']),'--source-sha',stage['validator']['sha256']],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=110)
 assert p.returncode==0,p.stdout[-1200:]+p.stderr[-2000:]
 result=json.loads(p.stdout.splitlines()[-1]);assert result['status']=='PUBLISHED_EPOCH_BOUNDARY_CPU_SHA_RAW_EMA_ADAM_TWO_RANK_RNG_ACCEPTED'and result['checkpoint']==held['checkpoint']and result['cursor']==held['cursor'];out['results'][str(layer)]=result
 (b/'walltime-boundary-CPU-accepted.json').write_text(json.dumps(result,sort_keys=True)+'\n')
print(json.dumps(out))
