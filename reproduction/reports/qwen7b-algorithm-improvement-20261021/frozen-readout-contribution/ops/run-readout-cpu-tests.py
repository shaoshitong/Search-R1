import json,hashlib,os,subprocess
from pathlib import Path
b=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/frozen-readout-question-prefix-diagnostic-v1');pub=json.loads((b/'source-publication.json').read_text());root=Path(pub['code_root'])
for name,item in pub['files'].items():
 p=root/name;assert p.stat().st_size==item['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256']
runtime=b.parent/'peer-newseed8192-g10-train-v1/code-d8f74420aa09';program="import types,sys,importlib,unittest; p=types.ModuleType('readout_overlay');p.__path__=["+repr(str(root))+"];sys.modules[p.__name__]=p; m=importlib.import_module('readout_overlay.test_readout_contribution'); result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(m));assert result.wasSuccessful()"
env=dict(os.environ,QWEN7B_CODE_ROOT=str(runtime),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
r=subprocess.run(['bash',str(runtime/'reproduction/operations/qwen7b_warmup/runtime.sh'),'-c',program],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True,timeout=45);p=b/'actual-cpu-tests.log';p.write_text(r.stdout);assert r.returncode==0,r.stdout[-2000:]
v=dict(status='REAL_REMOTE_FROZEN_READOUT_FOUR_CPU_TESTS_PASSED',source_commit=pub['commit'],log=dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()),models_loaded=False,optimizer_constructed=False);(b/'actual-cpu-tests.json').write_text(json.dumps(v,sort_keys=True));print(json.dumps(v))
