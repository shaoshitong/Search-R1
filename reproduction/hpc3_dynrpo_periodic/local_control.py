"""Existing HKUST3 campus SSH; private passwords travel only through stdin/RAM."""
import argparse
from pathlib import Path
import re
import shlex
import subprocess
import sys


def run(command,payload=b'',account='mou176'):
    if account not in ('mou176','jxu957'): raise ValueError('Unapproved HPC3 account')
    memory=Path('/Users/shaoshitong/Documents/remote_codex_control/memories/hpc3-credentials.private.md').read_text()
    line=next(x for x in memory.splitlines() if x.startswith('账号 '+account))
    password=re.split(r'密码\s*[:：]?\s*',line,1)[1].strip().strip('`')
    args=['sshpass','-e','ssh','-T','-oStrictHostKeyChecking=yes','-oControlMaster=no','-S','none',
        '-oHostKeyAlias=hpc3login.hpc.hkust-gz.edu.cn',
        '-oUserKnownHostsFile=/backup/shaoshitong/topd-hf-migration-20261009/hpc3-known-hosts',
        '-oConnectTimeout=10',account+'@10.120.81.30',command]
    bridge="import os,sys,subprocess; env=os.environ.copy(); env['SSHPASS']=sys.stdin.buffer.readline().decode().rstrip('\\n'); data=sys.stdin.buffer.read(); raise SystemExit(subprocess.run("+repr(args)+",input=data,env=env).returncode)"
    return subprocess.run(['ssh','hkust3','python3 -c '+shlex.quote(bridge)],
                          input=password.encode()+b'\n'+payload,capture_output=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--command',required=True)
    p.add_argument('--account',default='mou176',choices=('mou176','jxu957'));p.add_argument('--input-file')
    a=p.parse_args();payload=Path(a.input_file).read_bytes() if a.input_file else sys.stdin.buffer.read()
    r=run(a.command,payload,a.account);sys.stdout.buffer.write(r.stdout);sys.stderr.buffer.write(r.stderr)
    raise SystemExit(r.returncode)


if __name__=='__main__':
    main()
