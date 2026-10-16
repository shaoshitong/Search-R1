"""Explicit direct HPC3 control fallback; no Mac proxy or model data transport.

A caller must choose this adapter and bind the currently verified en0 address.
The password stays in RAM and a local controlling terminal, never argv/env/files.
Only bounded command-only controls and small metadata responses are supported.
Nonempty stdin payloads are explicitly rejected.
"""
import argparse
import errno
import ipaddress
import os
from pathlib import Path
import pty
import re
import select
import signal
import subprocess
import sys
import time

PRIVATE_MEMORY = Path('/Users/shaoshitong/Documents/remote_codex_control/memories/hpc3-credentials.private.md')
KNOWN_HOSTS = Path('/Users/shaoshitong/Documents/remote_codex_control/access/hpc3-20261008/known_hosts')
HOST_ALIAS = 'hpc3login.hpc.hkust-gz.edu.cn'
HOST = '10.120.81.30'
MAX_PAYLOAD = 1024*1024
MAX_OUTPUT = 8*1024*1024
PROMPT = re.compile(rb'password\s*:', re.IGNORECASE)


def password_for(account):
    if account not in ('mou176','jxu957'):
        raise ValueError('Unapproved HPC3 account')
    lines=PRIVATE_MEMORY.read_text().splitlines()
    matches=[line for line in lines if line.startswith('账号 '+account)]
    if len(matches)!=1:
        raise ValueError('Private account memory is missing or ambiguous')
    fields=re.split(r'密码\s*[:：]?\s*',matches[0],1)
    if len(fields)!=2:
        raise ValueError('Private account password field is missing')
    password=fields[1].strip().strip('`').encode()
    if not password or any(byte in password for byte in (0,10,13)):
        raise ValueError('Invalid private account password')
    return password


def command_args(command,account,source_address,interface='en0'):
    if account not in ('mou176','jxu957') or interface!='en0':
        raise ValueError('Only explicit approved HPC3 accounts and en0 are supported')
    address=str(ipaddress.IPv4Address(source_address))
    actual=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr',interface],text=True).strip()
    if actual!=address:
        raise ValueError('Explicit source address differs from current en0 address')
    if not KNOWN_HOSTS.is_file():
        raise ValueError('Existing HPC3 known_hosts is required')
    if not isinstance(command,str) or not command or '\0' in command or len(command.encode())>65536:
        raise ValueError('A bounded control command is required')
    return ['/usr/bin/ssh','-F','/dev/null','-T','-b',address,'-B',interface,
        '-oProxyCommand=none','-oProxyJump=none','-oStrictHostKeyChecking=yes',
        '-oHostKeyAlias='+HOST_ALIAS,'-oUserKnownHostsFile='+str(KNOWN_HOSTS),
        '-oGlobalKnownHostsFile=/dev/null','-oControlMaster=no','-S','none',
        '-oConnectTimeout=10','-oConnectionAttempts=1','-oServerAliveInterval=10',
        '-oServerAliveCountMax=2','-oNumberOfPasswordPrompts=1','-oBatchMode=no',
        '-oPreferredAuthentications=password,keyboard-interactive','-oPubkeyAuthentication=no',
        '-oGSSAPIAuthentication=no','-oHostbasedAuthentication=no',account+'@'+HOST,command]


def _execute(args,payload,password,*,timeout=90,prompt_timeout=30,output_limit=MAX_OUTPUT):
    """Command-only route using the same real terminal birth as interactive SSH.

    OpenSSH on macOS can refuse interactive authentication when only /dev/tty
    has a terminal but its standard descriptors are pipes. pty.fork makes all
    three descriptors real terminal descriptors, matching the proven route.
    """
    if payload:
        raise ValueError('Direct fallback accepts command-only small controls; stdin payload is unsupported')
    if timeout<=0 or not 0<prompt_timeout<=30:
        raise ValueError('Control timeout is outside the supported bound')
    pid,master=pty.fork()
    if pid==0:
        try:os.execv(args[0],args)
        except BaseException:os._exit(127)
    os.set_blocking(master,False)
    output=bytearray();authentication=bytearray();prompt_buffer=bytearray()
    sent=False;started=time.monotonic();failure=None;status=None
    def reap_nonblocking():
        nonlocal status
        if status is None:
            found,value=os.waitpid(pid,os.WNOHANG)
            if found:status=value
    def stop_child():
        nonlocal status
        reap_nonblocking()
        if status is not None:return
        os.kill(pid,signal.SIGTERM)
        until=time.monotonic()+3
        while status is None and time.monotonic()<until:
            reap_nonblocking()
            if status is None:time.sleep(.02)
        if status is None:
            os.kill(pid,signal.SIGKILL);_,status=os.waitpid(pid,0)
    try:
        while True:
            reap_nonblocking();now=time.monotonic()
            if now-started>=timeout:
                failure='Direct control command timed out';break
            if not sent and now-started>=prompt_timeout and status is None:
                failure='Direct SSH password prompt timed out';break
            ready,_,_=select.select([master],[],[],min(.2,max(.01,timeout-(now-started))))
            if not ready:
                if status is not None:break
                continue
            try:chunk=os.read(master,65536)
            except OSError as error:
                if error.errno==errno.EIO:break
                if error.errno in (errno.EAGAIN,errno.EWOULDBLOCK):continue
                raise
            if not chunk:break
            if sent:
                output.extend(chunk)
                if PROMPT.search(chunk):
                    failure='Direct SSH requested authentication more than once';break
            else:
                prompt_buffer.extend(chunk)
                match=PROMPT.search(prompt_buffer)
                if match:
                    authentication.extend(prompt_buffer[:match.start()])
                    authentication.extend(b'[SSH password prompt]\n')
                    output.extend(prompt_buffer[match.end():])
                    os.write(master,password+b'\n');sent=True;prompt_buffer.clear()
            if len(output)+len(authentication)+len(prompt_buffer)>output_limit:
                failure='Direct control output exceeded the small-metadata limit';break
        if not failure:
            while status is None and time.monotonic()-started<timeout:
                reap_nonblocking()
                if status is None:time.sleep(.02)
            if status is None:failure='Direct control command timed out'
        if failure:stop_child()
        redact=lambda value:bytes(value).replace(password,b'[REDACTED]')
        error=redact(authentication)
        if not sent:error+=redact(prompt_buffer)
        if failure:error+=b'\n'+failure.encode()+b'\n'
        return subprocess.CompletedProcess(args,124 if failure else os.waitstatus_to_exitcode(status),
            redact(output).replace(b'\r\n',b'\n'),error.replace(b'\r\n',b'\n'))
    finally:
        stop_child();os.close(master)


def run(command,payload=b'',*,account,source_address,interface='en0',timeout=90):
    args=command_args(command,account,source_address,interface)
    return _execute(args,payload,password_for(account),timeout=timeout)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--command',required=True)
    parser.add_argument('--account',required=True,choices=('mou176','jxu957'))
    parser.add_argument('--source-address',required=True)
    args=parser.parse_args()
    payload=sys.stdin.buffer.read(MAX_PAYLOAD+1)
    result=run(args.command,payload,account=args.account,source_address=args.source_address)
    sys.stdout.buffer.write(result.stdout);sys.stderr.buffer.write(result.stderr)
    raise SystemExit(result.returncode)


if __name__=='__main__':main()
