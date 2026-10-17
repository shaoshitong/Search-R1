"""Task-local direct management transport; never changes global SSH or training."""
import argparse
import datetime
import json
import os
from pathlib import Path
import shlex
import stat
import subprocess
import sys

MGMT = "import json,shlex,subprocess,datetime,sys\nfrom pathlib import Path\ncfg=json.loads(Path('/home/lichen/.claude/projects/-home-lichen/memory/reference_h100_retained18_route_20261008.public.json').read_text())\ntarget='h100-8';ip=cfg['targets'][target];gateway=cfg['gateway_ip'];user=cfg['user']\nprefix=['env','-u','LD_LIBRARY_PATH','ssh','-T','-i',cfg['management_identity_file'],'-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','-o','ConnectTimeout=8','-o','ConnectionAttempts=1']\nargs=prefix.copy()\nif ip!=gateway:args+=['-o','ProxyCommand='+shlex.join(prefix+['-W','%h:%p',user+'@'+gateway])]\nkeys=cfg.get('trusted_target_host_keys',{}).get(target)\nif keys:args+=['-o','KnownHostsCommand='+shlex.join(['/bin/echo','\\n'.join(keys)])]\nr=subprocess.run(args+[user+'@'+ip,WORKER_CMD],input=sys.stdin.buffer.read(),capture_output=True,timeout=240)\nprint(json.dumps({'route':'retained18 SSH-W h100-8','returncode':r.returncode,'stdout':r.stdout.decode(),'stderr':r.stderr.decode(),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}))\n"

def connection_options(connection):
    from .established_controller_transport import MODE, validate_connection
    mode = connection.get('transport_mode')
    if mode == MODE:
        validate_connection(connection)
        return dict(established_controller=True)
    if mode not in (None, 'direct_one_shot', 'direct_master', 'own_ordinary_alias_foreground_master'):
        raise ValueError('Unknown explicit management transport mode')
    return dict(control_path=connection.get('control_path'),
        compact_handshake=connection.get('compact_handshake', False))


def local_connection_path(root):
    work = Path(root)/'work/search-g1-prefix1024-diversity512-20261016'
    established = work/'ESTABLISHED_CONTROLLER_CURRENT.json'
    return established if established.is_file() else work/'MANAGEMENT_DIRECT_MASTER_CURRENT.json'


def remote(script, evidence, control_path=None, compat_handshake=False, target='h100-8', compact_handshake=False, *, established_controller=False):
    if established_controller:
        if control_path is not None or compat_handshake or compact_handshake:
            raise ValueError('Established controller must not be combined with socket or SSH-option overrides')
        from .established_controller_transport import remote as established_remote
        return established_remote(script, evidence, target=target)
    if target not in ('h100-5','h100-8','h100-10','h100-11'):
        raise ValueError('Only the four human-authorized same-cluster H100 nodes are supported')
    args=['env','-u','LD_LIBRARY_PATH','/usr/bin/ssh','-T',
          '-o','BatchMode=yes','-o','IdentitiesOnly=yes',
          '-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no',
          '-o','ConnectTimeout=30','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=4']
    # Explicit task-local opt-in; no fallback, reconnect loop, or global SSH edit.
    # This was read back against the existing mgmt-host host key and identity.
    if compact_handshake:
        args+=['-o','IPQoS=none','-o','KexAlgorithms=curve25519-sha256',
               '-o','HostKeyAlgorithms=ssh-ed25519','-o','Ciphers=aes128-gcm@openssh.com',
               '-o','MACs=hmac-sha2-256','-o','Compression=no']
        if control_path is None:
            args+=['-o','ControlMaster=no','-o','ControlPath=none']
    elif compat_handshake:
        args+=['-o','IPQoS=none','-o','KexAlgorithms=curve25519-sha256']
    source_ip=os.environ.get('G1_MGMT_SOURCE_IP')
    if source_ip:
        import ipaddress
        ipaddress.IPv4Address(source_ip)
        actual=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip()
        if source_ip != actual:
            raise ValueError('Source IP must match the currently observed en0 address')
        proxy=Path(__file__).resolve().with_name('source_bound_proxy.py')
        args+=['-o','ProxyCommand='+shlex.join([sys.executable,str(proxy),'--source',source_ip,
                '--host','%h','--port','%p']),'-o','IPQoS=none','-o','KexAlgorithms=curve25519-sha256']
    if control_path:
        socket=Path(control_path);parent=socket.parent
        if not parent.name.startswith('qprobe3-mgmt-'):
            raise ValueError('Only this task management socket namespace is accepted')
        info=parent.stat();sock=socket.lstat()
        if info.st_uid!=os.getuid() or stat.S_IMODE(info.st_mode)!=0o700 or sock.st_uid!=os.getuid() or not stat.S_ISSOCK(sock.st_mode):
            raise ValueError('Management control socket must be owned and private')
        subprocess.run(['/usr/bin/ssh','-S',str(socket),'-O','check','mgmt-host'],check=True,capture_output=True,timeout=10)
        args+=['-S',str(socket),'-o','ControlMaster=no']
    # Keep source payloads off nested shell argv (Linux MAX_ARG_STRLEN).
    # The existing management relay forwards stdin unchanged to this Python.
    command='/usr/bin/python3 -'
    management=MGMT.replace("target='h100-8'",'target='+repr(target)).replace('retained18 SSH-W h100-8','retained18 SSH-W '+target).replace('WORKER_CMD',repr(command))
    args+=['mgmt-host','env -u LD_LIBRARY_PATH /usr/bin/python3 -c '+shlex.quote(management)]
    process=subprocess.run(args,input=script.encode(),capture_output=True,timeout=270)
    record=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),returncode=process.returncode,
                stdout=process.stdout.decode(),stderr=process.stderr.decode(),route='mgmt-host alias -> retained18 -> '+target,
                global_network_configuration_changed=False)
    with Path(evidence).open('x') as stream:json.dump(record,stream,indent=2)
    if process.returncode:raise RuntimeError(record['stderr'])
    result=json.loads(record['stdout'])
    if result['returncode']:raise RuntimeError(result['stderr'])
    return result['stdout']


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--evidence',required=True);parser.add_argument('--control-path');parser.add_argument('--compat-handshake',action='store_true');parser.add_argument('--compact-handshake',action='store_true');parser.add_argument('--target',choices=['h100-5','h100-8','h100-10','h100-11'],default='h100-8')
    args=parser.parse_args();print(remote(sys.stdin.read(),args.evidence,args.control_path,args.compat_handshake,args.target,args.compact_handshake))


if __name__=='__main__':main()
