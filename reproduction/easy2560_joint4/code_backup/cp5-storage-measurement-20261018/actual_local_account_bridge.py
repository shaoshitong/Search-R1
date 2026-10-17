"""Local HKUST3 bridge; credentials are read in RAM and sent only over stdin.

The existing authorized campus transport module is supplied explicitly. Never
print credentials or put them in arguments, receipts, source archives or logs.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import re
import shlex
import sys
import subprocess


LOGIN_HOSTS = ('10.120.81.30', '10.120.48.31', '10.120.48.32', '10.120.48.33')
LOGIN_NAMES = dict(zip(LOGIN_HOSTS[1:], ('ACD-Login-1', 'ACD-Login-2', 'ACD-Login-3')))


def bound_command(command, login_host, expected_host_identity):
    """Fail before the requested command when a pinned kernel identity differs."""
    if expected_host_identity is None:
        return command
    expected = expected_host_identity
    if (not isinstance(expected, dict) or set(expected) != {'hostname', 'boot_id', 'pid_namespace'}
            or login_host not in LOGIN_NAMES or expected['hostname'] != LOGIN_NAMES[login_host]
            or not isinstance(expected['boot_id'], str)
            or re.fullmatch(r'[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}', expected['boot_id']) is None
            or not isinstance(expected['pid_namespace'], str)
            or re.fullmatch(r'pid:\[[0-9]+\]', expected['pid_namespace']) is None):
        raise ValueError('Explicit observed login-node kernel identity required')
    code = ('import json,os,socket,sys\n'
            'expected=json.loads('+repr(json.dumps(expected, sort_keys=True))+')\n'
            "with open('/proc/sys/kernel/random/boot_id') as f: boot=f.read().strip()\n"
            "actual=dict(hostname=socket.gethostname(),boot_id=boot,pid_namespace=os.readlink('/proc/self/ns/pid'))\n"
            'if actual!=expected:\n'
            " print(json.dumps(dict(status='UNKNOWN_HOST_IDENTITY_MISMATCH',actual=actual,expected=expected)),file=sys.stderr)\n"
            ' sys.exit(86)\n'
            "os.execv('/bin/bash',['bash','-c',"+repr(command)+'])\n')
    return 'python3 -c '+shlex.quote(code)


def _campus_long(command,payload,timeout):
    """Same two authorized HK3 routes; both preserve an explicit long deadline."""
    primary=['ssh','-T','-oBatchMode=yes','-oStrictHostKeyChecking=yes','-oConnectTimeout=8','hkust3',command]
    fallback=['ssh','-T','-B','en0','-oProxyCommand=none','-oProxyJump=none',
        '-oBatchMode=yes','-oStrictHostKeyChecking=yes','-oConnectTimeout=8','hkust3',command]
    for index,args in enumerate((primary,fallback)):
        try:
            result=subprocess.run(args,input=payload,capture_output=True,text=True,timeout=timeout)
        except subprocess.TimeoutExpired as error:
            output=error.stdout or ''
            if isinstance(output,bytes):output=output.decode(errors='replace')
            return subprocess.CompletedProcess(args,124,output,
                'Observation timed out; inspect the existing remote operation before retry')
        if result.returncode!=255 or index==1:return result


def run(account, command, payload, control_work, shell=False, login_host=LOGIN_HOSTS[0],
        expected_host_identity=None, observation_timeout=55):
    if type(observation_timeout) is not int or not 1<=observation_timeout<=1800:
        raise ValueError('Explicit bounded observation timeout must be an integer1..1800')
    if account not in ('mou176', 'jxu957'):
        raise ValueError('Account is outside the explicit authorized pair')
    if login_host not in LOGIN_HOSTS:
        raise ValueError('Login host is outside the observed HPC3 gateway/login nodes')
    command = bound_command(command, login_host, expected_host_identity)
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
    code = "import os,sys,subprocess;env=os.environ.copy();env['SSHPASS']=sys.stdin.readline().rstrip('\\n');data=sys.stdin.read();r=subprocess.run("+repr(ssh)+",input=data,env=env,capture_output=True,text=True,timeout="+str(observation_timeout)+");print(r.stdout,end='');print(r.stderr,file=sys.stderr,end='');sys.exit(r.returncode)"
    outer_command='python3 -c '+shlex.quote(code)
    result = (m.campus('hkust3',outer_command,input=password+'\n'+payload,observation_timeout=65)
              if observation_timeout==55 else
              _campus_long(outer_command,password+'\n'+payload,observation_timeout+10))
    password = None
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--account', required=True, choices=('mou176', 'jxu957'))
    p.add_argument('--control-work', required=True); p.add_argument('--command', required=True)
    p.add_argument('--shell', action='store_true')
    p.add_argument('--login-host', choices=LOGIN_HOSTS, default=LOGIN_HOSTS[0],
                   help='Explicit HPC3 endpoint; strict existing host-key verification is preserved')
    p.add_argument('--host-identity-json', type=json.loads,
                   help='Previously observed hostname, boot_id and pid_namespace; mismatch never runs the command')
    p.add_argument('--observation-timeout',type=int,default=55,help='Bounded SSH observation only; a timeout never authorizes replay')
    a = p.parse_args(); result = run(a.account, a.command, sys.stdin.read(), a.control_work,
                                    shell=a.shell, login_host=a.login_host,
                                    expected_host_identity=a.host_identity_json, observation_timeout=a.observation_timeout)
    print(result.stdout, end=''); print(result.stderr, file=sys.stderr, end=''); sys.exit(result.returncode)
