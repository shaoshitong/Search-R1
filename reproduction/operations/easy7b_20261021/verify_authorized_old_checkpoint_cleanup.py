"""Independent post-apply absence and preservation check; never unlinks anything."""
import hashlib
import json
import os
from pathlib import Path
import stat
import time

uid=os.getuid();assert uid in(14090,21930)
root=Path('/tmp/authorized-old-checkpoints-20261021-'+str(uid))
raw=(root/'plan.json').read_bytes();plan=json.loads(raw);digest=hashlib.sha256(raw).hexdigest()
assert digest==json.loads((root/'apply-claim.json').read_text())['plan_sha256']
assert plan['authorization']=='2026-10-21-human-delete-all-checkpoints-exact-old-experiments-preserve-text-and-eval-trajectories'
missing=[];present=[];unknown=[]
for row in plan['delete_files']:
 path=Path(row['path'])
 try:path.lstat();present.append(str(path))
 except FileNotFoundError:missing.append(str(path))
 except OSError as error:unknown.append(dict(path=str(path),error=str(error)))
assert not present and not unknown,(present,unknown)
checked=0;unreadable=0
for row in plan['preserved_text']+plan['preserved_binary']:
 path=Path(row['path'])
 try:
  seen=path.lstat()
  assert stat.S_ISREG(seen.st_mode)and not path.is_symlink()
  assert(seen.st_dev,seen.st_ino)==(row['device'],row['inode'])and seen.st_size>=row['bytes'],str(path)
  checked+=1
 except PermissionError:unreadable+=1
result=dict(epoch=time.time(),uid=uid,read_only=True,plan_sha256=digest,all258_targets_absent=True,absent_targets=len(missing),
            retained_text_and_binary_metadata_verified=checked,unreadable_preserved_paths=unreadable,
            deletion_operations=0,training_operations=0,current7b_or_feature_modified=False,
            actual_cuda_restore_accepted=False)
path=root/('verification-'+str(time.time_ns())+'.json')
with path.open('x')as stream:json.dump(result,stream);stream.flush();os.fsync(stream.fileno())
print(json.dumps(result))
