"""Local HKUST3 bridge; credentials are read in RAM and sent only over stdin.

The existing authorized campus transport module is supplied explicitly. Never
print credentials or put them in arguments, receipts, source archives or logs.
"""
import argparse
import importlib.util
from pathlib import Path
import re
import shlex
import sys


LOGIN_HOSTS = ('10.120.81.30', '10.120.48.31', '10.120.48.32', '10.120.48.33')


def run(account, command, payload, control_work, shell=False, login_host=LOGIN_HOSTS[0]):
    if account not in ('mou176', 'jxu957'):
        raise ValueError('Account is outside the explicit authorized pair')
    if login_host not in LOGIN_HOSTS:
        raise ValueError('Login host is outside the observed HPC3 gateway/login nodes')
    spec = importlib.util.spec_from_file_location('campus_transport', Path(control_work)/'start_cosearch_flash_hf.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    secret = Path('/Users/shaoshitong/Documents/remote_codex_control/memories/hpc3-credentials.private.md').read_text()
    line = next(l for l in secret.splitlines() if l.startswith('账号 '+account))
    password = re.split(r'密码\s*[:：]?\s*', line, 1)[1].strip().strip('`')
    secret = line = None
    ssh = ['sshpass', '-e', 'ssh', '-T', '-oStrictHostKeyChecking=yes',
           '-oControlMaster=no', '-S', 'none',
           '-oHostKeyAlias=hpc3login.hpc.hkust-gz.edu.cn',
           '-oUserKnownHostsFile=/backup/shaoshitong/topd-hf-migration-20261009/hpc3-known-hosts',
           '-oConnectTimeout=8', account+'@'+login_host]
    if shell:
        if payload:
            raise ValueError('Shell channel requires a self-contained command')
        # Request the authenticated account's normal shell instead of an exec
        # channel. This changes only the SSH channel type, not owner/privilege.
        payload = command+'\n_qprobe_bridge_status=$?\nexit $_qprobe_bridge_status\n'
    else:
        ssh.append(command)
    code = "import os,sys,subprocess;env=os.environ.copy();env['SSHPASS']=sys.stdin.readline().rstrip('\\n');data=sys.stdin.read();r=subprocess.run("+repr(ssh)+",input=data,env=env,capture_output=True,text=True,timeout=55);print(r.stdout,end='');print(r.stderr,file=sys.stderr,end='');sys.exit(r.returncode)"
    result = m.campus('hkust3', 'python3 -c '+shlex.quote(code), input=password+'\n'+payload, observation_timeout=65)
    password = None
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--account', required=True, choices=('mou176', 'jxu957'))
    p.add_argument('--control-work', required=True); p.add_argument('--command', required=True)
    p.add_argument('--shell', action='store_true')
    p.add_argument('--login-host', choices=LOGIN_HOSTS, default=LOGIN_HOSTS[0],
                   help='Explicit HPC3 endpoint; strict existing host-key verification is preserved')
    a = p.parse_args(); result = run(a.account, a.command, sys.stdin.read(), a.control_work,
                                    shell=a.shell, login_host=a.login_host)
    print(result.stdout, end=''); print(result.stderr, file=sys.stderr, end=''); sys.exit(result.returncode)
