import hashlib,json,subprocess,time
from pathlib import Path
A=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021');c=A/'full169028-Z-seed2-terminal-CPU-acceptance-20261023-v1/code-0f051920d4de';j=json.loads((c/'staged.json').read_bytes())
def proof(p):
 raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
assert proof(Path(j['validator']['path']))==j['validator']
for e in j['entries']:
 assert proof(Path(e['entry']['path']))==e['entry'];assert proof(Path(e['actual_training_source_publication']['path']))==e['actual_training_source_publication'];assert proof(Path(e['actual_module']['path']))==e['actual_module'];helptext=subprocess.check_output(['bash',e['entry']['path'],'--help'],universal_newlines=True,stderr=subprocess.STDOUT,timeout=20);assert '--base BASE'in helptext and '--epoch'in helptext
print(json.dumps(dict(status='INDEPENDENT_STAGED_SOURCE_ENTRY_AND_TRUE_CLI_HELP_VERIFIED',time=time.time(),source_commit=j['source_commit'],validator=j['validator'],entry=j['entries'][0]['entry'],actual_cli_help=helptext,checkpoint_or_tensor_read=False,Slurm_changed=False,acceptor_started=False,final4096_read=False)))
