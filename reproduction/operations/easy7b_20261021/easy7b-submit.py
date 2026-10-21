import os,sys,json,time
from pathlib import Path
base=Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-20261021');commit='d6690e86152cbd6f9400317646de04d9f36ac2e1'
sys.path.insert(0,str(base/'source'/commit))
from reproduction.easy2560_joint4.control import submit
print(json.dumps(dict(epoch=time.time(),submission=submit(base,commit,actor_micro=8))),flush=True)
