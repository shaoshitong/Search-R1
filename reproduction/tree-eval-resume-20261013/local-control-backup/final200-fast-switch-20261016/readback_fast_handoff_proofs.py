import sys,os,json,hashlib,shlex,time
from pathlib import Path
sys.path.insert(0,'work/tree-speed-repair-20261008');import ssh_retained18 as ssh
os.environ['TREE_MGMT_CONTROL_PATH']='/tmp/tree-mgmt-3fw9a1ab/control'
source=r'''import json,hashlib,time
from pathlib import Path
r=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-nested-branch-fresh200-not-slurm;systemd:778b2b43dad90fc0c1c5d95d77c2326286011b73/output/final-eval-fast')
o={}
for n in ['archive-proof.json','asset-manifest.json','original-terminal.json','handoff-authorized.json','dispatch-once-tp1-20261016.json']:
 b=(r/n).read_bytes();assert len(b)<65536;o[n]=dict(path=str(r/n),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),raw=b.decode())
print(json.dumps(dict(epoch=time.time(),files=o)))'''
a=ssh.run('python3 -c '+shlex.quote(source),timeout=60)
if a.returncode:raise RuntimeError(a.stderr)
x=json.loads(a.stdout);d=Path('outputs/tree-nested-branch-handoff-20261013/final200-fast-handoff-proofs');d.mkdir(exist_ok=True)
for n,v in x['files'].items():
 b=v.pop('raw').encode();assert len(b)==v['bytes'] and hashlib.sha256(b).hexdigest()==v['sha256'];p=d/n
 if p.exists():assert p.read_bytes()==b
 else:p.write_bytes(b)
 for _ in range(2):assert p.read_bytes()==b
p=Path('outputs/tree-nested-branch-handoff-20261013/FINAL200_FAST_HANDOFF_PROOF_READBACK.json');assert not p.exists();p.write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x))
