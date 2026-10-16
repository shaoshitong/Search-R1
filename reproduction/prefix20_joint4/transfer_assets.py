"""HKUST2→HKUST3 selected probe parts; source stays until final-hop acceptance."""
import fcntl
import json
from pathlib import Path
import shlex
import socket
import subprocess
import time

from reproduction.prefix20_joint4.assets import sha,atomic

SOURCE=Path('/data/shaoshitong/qprobe-prefix20-hpc3-4gpu-20261016/assets')
TARGET='/backup/shaoshitong/qprobe-prefix20-hpc3-4gpu-20261016/assets'
PEER='shaoshitong@10.120.17.122'
SSH=['ssh','-T','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=10',PEER]


def remote(code):
    p=subprocess.run(SSH+['python3 -c '+shlex.quote(code)],capture_output=True,text=True,timeout=90)
    if p.returncode:raise RuntimeError('HKUST3_ASSET_ACTION_FAILED; preserve owned claim and partials')
    return json.loads(p.stdout)


def main():
    if socket.gethostname()!='xielab2':raise ValueError('FORWARD_MUST_RUN_ON_HKUST2')
    with (SOURCE/'forward-hkust3.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=json.loads((SOURCE/'probe-asset-plan.json').read_text())
        complete=json.loads((SOURCE/'complete.json').read_text())
        if complete['status']!='PROBE_ALL_MEMBERS_FRESH_SHA_VERIFIED':raise ValueError('SOURCE_PROBE_NOT_VERIFIED')
        for part in plan['archive']['parts']:
            p=SOURCE/part['name']
            if p.stat().st_size!=part['bytes'] or sha(p)!=part['sha256']:
                raise ValueError('SOURCE_PART_CHANGED_BEFORE_FORWARD')
        target=repr(TARGET)
        preflight=remote("import json,os,socket,time;from pathlib import Path;r=Path("+target+");r.mkdir(parents=True,exist_ok=True);s=os.statvfs(r);assert socket.gethostname()=='xielab3' and s.f_bavail*s.f_frsize>5*1024**3;p=r/'forward-claim.json';f=p.open('x');json.dump(dict(created_unix=time.time(),source_host='xielab2',source_root="+repr(str(SOURCE))+"),f);f.close();print(json.dumps(dict(host=socket.gethostname(),available_bytes=s.f_bavail*s.f_frsize,claim=str(p))))")
        atomic(SOURCE/'forward-hkust3-start.json',dict(pid=__import__('os').getpid(),started_unix=time.time(),target_preflight=preflight))
        transport='ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=10'
        paths=[str(SOURCE/p['name']) for p in plan['archive']['parts']]+[str(SOURCE/'assets.py'),str(SOURCE/'probe-asset-plan.json')]
        subprocess.run(['rsync','-a','--partial','--append-verify','--no-owner','--no-group','-e',transport,*paths,PEER+':'+TARGET+'/'],check=True,timeout=900)
        code="import sys,json;from pathlib import Path;r=Path("+target+");sys.path.insert(0,str(r));from assets import sha,unpack,atomic;p=json.loads((r/'probe-asset-plan.json').read_text());assert all((r/x['name']).stat().st_size==x['bytes'] and sha(r/x['name'])==x['sha256'] for x in p['archive']['parts']);proof=unpack(r,p['archive'],p['member_manifest']);proof.update(repo=p['repo'],revision=p['revision'],route_stage='HKUST3_FORWARD_ALL_PARTS_AND_MEMBERS_VERIFIED');atomic(r/'complete.json',proof);print(json.dumps(proof))"
        proof=remote(code)
        atomic(SOURCE/'hkust3-verified.json',dict(proof,completed_unix=time.time(),actual_route='HF→HKUST2→HKUST3',hpc3_acceptance_pending=True))


if __name__=='__main__':main()
