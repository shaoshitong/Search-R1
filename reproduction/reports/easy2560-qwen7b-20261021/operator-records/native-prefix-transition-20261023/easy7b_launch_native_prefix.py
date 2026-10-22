import json,sys,subprocess,time
from pathlib import Path
sys.path.insert(0,'/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fastpath-probe-20261022/source/ab8c44c8b68d912399b77b354fee5a832b5e9f9a')
from reproduction.operations.easy7b_20261021.launch_lean_transition_on_retrieval import launch
from reproduction.operations.easy7b_20261021.fa3_lean_transition import source_binding
base=Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fa3-nativeprefix-r1-20261022')
a=base/'preparation/native-prefix-transition-qualified-authorization.json'
source='/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fa3-lean-r1-20261022'
with source_binding(source_base=source,source_job='749255',source_commit='48b87a7399a25f78b4fcf077e946a5a2b75ecace',authorization=json.loads(a.read_text()),minimum_outer=54) as bound:
 terminal=bound._terminal_source()
result=launch(str(base),'c7976da3afc863a1c20b99daa7cc99de3bb61f15',str(a),allocation_job='745198',source_base=source,source_job='749255',source_commit='48b87a7399a25f78b4fcf077e946a5a2b75ecace',minimum_outer=54)
print(json.dumps({'epoch':time.time(),'source_terminal':terminal,'launch':result}),flush=True)
