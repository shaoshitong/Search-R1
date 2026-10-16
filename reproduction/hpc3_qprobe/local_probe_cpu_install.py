"""Target networking failed: transfer pinned public wheels in RAM only.

No wheel/cache/environment file is created on the Mac. Credentials use the
existing stdin bridge. Package bytes are installed solely in this remote task.
"""
import argparse
import base64
import hashlib
import json
import shlex
import subprocess
import urllib.request
from reproduction.hpc3_qprobe.local_account_bridge import run
from reproduction.hpc3_qprobe.probe_cpu_runtime import lock,root

REMOTE=r'''import base64,hashlib,io,json,os,sys,time,zipfile
from pathlib import Path
v=json.load(sys.stdin);target=Path(v['target']);expected='/data/user/jxu957/experiments/qprobe-teacher-warmup200-20261014/runtime/probe-cpu-'
assert str(target).startswith(expected) and os.getuid()==14090
os.umask(0o007);partial=target.with_suffix('.partial');target.parent.mkdir(parents=True,exist_ok=True)
assert not target.exists() and not partial.exists(),'Inspect existing dependency installation rather than overwrite'
partial.mkdir();site=partial/'site-packages';site.mkdir();files={}
for name,record in v['wheels'].items():
 raw=base64.b64decode(v['payloads'][name]);assert len(raw)==record['size'] and hashlib.sha256(raw).hexdigest()==record['sha256']
 with zipfile.ZipFile(io.BytesIO(raw)) as z:
  for m in z.infolist():
   n=m.filename;assert not n.startswith('/') and '..' not in Path(n).parts
   assert not ((m.external_attr>>16)&0o170000)==0o120000
   if m.is_dir():continue
   assert n not in files,'Dependency file collision'
   b=z.read(m);p=site/n;p.parent.mkdir(parents=True,exist_ok=True)
   with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
   files[n]=hashlib.sha256(b).hexdigest()
for n,h in files.items():assert hashlib.sha256((site/n).read_bytes()).hexdigest()==h
out=dict(status='CPU_PROBE_WHEELS_FULL_SHA_PASS',wheels=v['wheels'],files=files,created_unix=time.time(),source_commit=v['commit'],shared_runtime_modified=False,local_disk_bytes=0)
with (partial/'manifest.json').open('x') as f:json.dump(out,f,indent=2);f.flush();os.fsync(f.fileno())
partial.rename(target)
print(json.dumps(dict(status=out['status'],target=str(target),file_count=len(files),wheel_bytes=sum(x['size'] for x in v['wheels'].values()),source_commit=v['commit'])))
'''


def main():
    p=argparse.ArgumentParser();p.add_argument('--control-work',required=True);a=p.parse_args()
    sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    remote=subprocess.check_output(['git','ls-remote','git@github.com:shaoshitong/Search-R1.git','refs/heads/codex/qprobe-teacher-warmup-hpc3-20261014'],text=True).split()[0]
    assert sha==remote,'Publish installer before external execution'
    wheels=lock();payloads={}
    for name,record in wheels.items():
        with urllib.request.urlopen(record['url'],timeout=30) as response:b=response.read()
        assert len(b)==record['size'] and hashlib.sha256(b).hexdigest()==record['sha256']
        payloads[name]=base64.b64encode(b).decode()
    value=dict(target=str(root()),wheels=wheels,payloads=payloads,commit=sha)
    r=run('jxu957','python3 -c '+shlex.quote(REMOTE),json.dumps(value),a.control_work)
    assert r.returncode==0,r.stderr
    print(r.stdout,end='')


if __name__=='__main__':main()
