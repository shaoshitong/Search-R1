import sys,os,json,subprocess,time
from pathlib import Path
b=Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-20261021');root=b/'source'/'d6690e86152cbd6f9400317646de04d9f36ac2e1'
sys.path.insert(0,str(root))
from reproduction.easy2560_joint4.control import bootstrap
proof=bootstrap(b,'d6690e86152cbd6f9400317646de04d9f36ac2e1')
env=dict(os.environ,PYTHONPATH=str(root),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
x=subprocess.run([str(b/'env/bin/python'),'-B','-m','unittest','reproduction.easy2560_joint4.test_model7b','reproduction.common8.test_configuration'],cwd=root,env=env,capture_output=True,text=True,timeout=40)
print(json.dumps(dict(epoch=time.time(),bootstrap=proof,tests_returncode=x.returncode,tests_stdout=x.stdout,tests_stderr=x.stderr,training_submitted=False)))
