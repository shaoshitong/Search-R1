import shlex,subprocess,json
from pathlib import Path
from reproduction.hpc3_prefix5.direct_control import run
from reproduction.hpc3_prefix5.control import PYTHON,OUTPUT
code=r'''
import os,json,time,gc
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
b=Path('/data/user/jxu957/experiments/dynrpo-prefix5-thread5-20261017-r3');c=json.loads((b/'preparation/launch-config.json').read_text())
s=torch.load(c['probe_source_checkpoint'],map_location='cpu',weights_only=False,mmap=True)
def scalar(v):
 if isinstance(v,dict):return {k:scalar(x) for k,x in v.items() if isinstance(x,(dict,int,float,bool))}
 if isinstance(v,(int,float,bool)):return v
 return None
binding={k:v for k,v in s.get('binding',{}).items() if any(term in k for term in ['feature','question','source','bank','manifest','split','seed','group']) and isinstance(v,(str,int,float,bool,dict))}
manifest_path=Path(c['fixed_test_features'])/'manifest.json';feature=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
composite=json.loads(Path(c['composite_manifest_path']).read_text());ident=json.loads(Path(c['fixed_test_identity_path']).read_text())
print(json.dumps(dict(epoch=time.time(),source_epoch=s.get('epoch'),metrics=scalar(s.get('epoch_metrics',{})),binding=binding,feature_manifest_keys=list(feature),feature_manifest_small={k:v for k,v in feature.items() if isinstance(v,(str,int,float,bool))},composite_keys=list(composite),composite_small={k:v for k,v in composite.items() if isinstance(v,(str,int,float,bool))},composite_source_summary=[{k:v for k,v in r.items() if isinstance(v,(str,int,float,bool))} for r in composite.get('sources',[])[:2]],fixed_identity_source=ident.get('source_identity'),tensor_payloads_accessed=False,cuda_initialized=torch.cuda.is_initialized())))
'''
a=subprocess.check_output(['ipconfig','getifaddr','en0'],text=True).strip();r=run(PYTHON+' -c '+shlex.quote(code),account='mou176',source_address=a,timeout=25)
if r.returncode:raise RuntimeError(r.stdout.decode()+r.stderr.decode())
v=next(json.loads(line) for line in reversed(r.stdout.decode().splitlines()) if line.startswith('{'));(Path(OUTPUT)/'probe-source-ema5-warmup-scalar-metrics.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v))
