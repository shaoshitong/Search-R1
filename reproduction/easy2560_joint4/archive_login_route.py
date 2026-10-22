"""Explicit pinned SSH routing for newly launched Easy archive consumers.

No environment overrides, automatic fallback, credentials, or network calls.
The default argv remains identical to the frozen Login1 supervisor transport.
This module has no package imports so a companion can load it privately while
its archive implementation continues to import the frozen083 package.
"""
import json
import re
import shlex


DEFAULT_LOGIN_HOST = '10.120.48.31'
DEFAULT_LOGIN_IDENTITY = dict(hostname='ACD-Login-1',
    boot_id='a27512a3-24a0-4ca3-aa92-98d2a5256d9a', pid_namespace='pid:[4026531836]')
LOGIN_HOSTS = {'10.120.48.31': 'ACD-Login-1', '10.120.48.32': 'ACD-Login-2'}
KNOWN_HOSTS = '/backup/shaoshitong/topd-hf-migration-20261009/hpc3-known-hosts'
HOST_KEY_ALIAS = 'hpc3login.hpc.hkust-gz.edu.cn'


def resolve_login_route(login_host=None, login_identity_json=None):
    host = DEFAULT_LOGIN_HOST if login_host is None else login_host
    if not isinstance(host, str) or host not in LOGIN_HOSTS:
        raise ValueError('EASY_ARCHIVE_EXPLICIT_LOGIN_HOST_REQUIRED')
    if login_identity_json is None:
        if host != DEFAULT_LOGIN_HOST:
            raise ValueError('EASY_ARCHIVE_NONDEFAULT_LOGIN_REQUIRES_EXPLICIT_IDENTITY')
        identity = dict(DEFAULT_LOGIN_IDENTITY)
    else:
        identity = json.loads(login_identity_json)
    if (not isinstance(identity, dict) or set(identity) != set(DEFAULT_LOGIN_IDENTITY) or
            identity.get('hostname') != LOGIN_HOSTS[host] or
            not isinstance(identity.get('boot_id'), str) or
            re.fullmatch(r'[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}', identity['boot_id']) is None or
            not isinstance(identity.get('pid_namespace'), str) or
            re.fullmatch(r'pid:\[[0-9]+\]', identity['pid_namespace']) is None):
        raise ValueError('EASY_ARCHIVE_COMPLETE_LOGIN_IDENTITY_REQUIRED')
    return host, dict(identity)


def route_record(login_host=None, login_identity_json=None):
    host, identity = resolve_login_route(login_host, login_identity_json)
    return dict(ssh_peer='jxu957@'+host, hpc_identity=identity,
                known_hosts=KNOWN_HOSTS, host_key_alias=HOST_KEY_ALIAS,
                automatic_fallback=False)


def bound_command(command, identity):
    code = ('import json,os,socket,sys\nexpected=json.loads(' + repr(json.dumps(identity, sort_keys=True)) + ')\n'
        "with open('/proc/sys/kernel/random/boot_id') as f: boot=f.read().strip()\n"
        "actual=dict(hostname=socket.gethostname(),boot_id=boot,pid_namespace=os.readlink('/proc/self/ns/pid'))\n"
        'if actual!=expected:\n'
        " print(json.dumps(dict(status='UNKNOWN_HOST_IDENTITY_MISMATCH',actual=actual,expected=expected)),file=sys.stderr)\n"
        ' sys.exit(86)\n'
        "os.execv('/bin/bash',['bash','-c'," + repr(command) + '])\n')
    return 'python3 -B -c ' + shlex.quote(code)


def ssh_argv(command, *, login_host=None, login_identity_json=None):
    host, identity = resolve_login_route(login_host, login_identity_json)
    return ['sshpass', '-e', 'ssh', '-T', '-oStrictHostKeyChecking=yes', '-oControlMaster=no', '-S', 'none',
        '-oHostKeyAlias='+HOST_KEY_ALIAS, '-oUserKnownHostsFile='+KNOWN_HOSTS,
        '-oConnectTimeout=8', '-oServerAliveInterval=15', '-oServerAliveCountMax=3',
        'jxu957@'+host, bound_command(command, identity)]
