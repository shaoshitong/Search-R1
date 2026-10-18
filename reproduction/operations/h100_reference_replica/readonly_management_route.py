"""Task-only, authenticated read-only fallback to the same h100-11 instance.

The primary mgmt-host -> retained18 route is unchanged. The optional fallback
uses known5 and the NodeAddr published by Slurm, with the original14 host key
as HostKeyAlias. Never use automatic retries for submit, cancel or once writers.
"""
import shlex
import subprocess

MODES = ('retained18', 'lan5', 'retained18_fallback_lan5')


def lan5_argv(ctl, target, command):
    if target != 'h100-11':
        raise ValueError('Only the actually verified same h100-11 LAN endpoint is supported')
    c = ctl.H100_ACCESS
    if c['targets'][target] != '10.33.4.14' or c['targets']['h100-5'] != '10.33.0.5':
        raise ValueError('Known target/gateway mapping changed; qualify it before reuse')
    keys = c.get('trusted_target_host_keys', {}).get(target)
    if not keys:
        raise ValueError('Original h100-11 host keys are required; never accept a new key')
    prefix = ['env','-u','LD_LIBRARY_PATH','ssh','-T','-i',c['management_identity_file'],
              '-o','IdentitiesOnly=yes','-o','BatchMode=yes',
              '-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no',
              '-o','ConnectTimeout=10','-o','ConnectionAttempts=1']
    gateway = prefix.copy()
    gateway_keys = c.get('trusted_target_host_keys', {}).get('h100-5')
    if gateway_keys:
        gateway += ['-o','KnownHostsCommand='+shlex.join(['/bin/echo','\n'.join(gateway_keys)])]
    proxy = shlex.join(gateway+['-W','%h:%p',c['user']+'@10.33.0.5'])
    client = prefix+['-o','HostKeyAlias=10.33.4.14','-o','ProxyCommand='+proxy,
                     '-o','KnownHostsCommand='+shlex.join(['/bin/echo','\n'.join(keys)]),
                     c['user']+'@10.32.0.11',command]
    return ctl.ssh('mgmt-host')+[shlex.join(client)]


def run_readonly(ctl, target, command, *, input_text, timeout, mode='retained18'):
    if mode not in MODES:
        raise ValueError('Unknown explicitly selected observation transport')
    attempts = [('retained18',ctl.route(target,command))]
    if mode == 'lan5':
        attempts = [('lan5',lan5_argv(ctl,target,command))]
    elif mode == 'retained18_fallback_lan5':
        attempts.append(('lan5',lan5_argv(ctl,target,command)))
    failures = []
    for name, argv in attempts:
        try:
            response = subprocess.run(argv,input=input_text,text=True,capture_output=True,timeout=timeout)
        except subprocess.TimeoutExpired:
            failures.append(dict(route=name,timeout=True))
            if name == attempts[-1][0]:
                raise
            continue
        if response.returncode == 255 and name != attempts[-1][0]:
            failures.append(dict(route=name,returncode=255,stderr=response.stderr[-2000:]))
            continue
        # Remote Python/report failures must be surfaced, not silently replayed.
        return response, dict(route=name,earlier_transport_failures=failures,
                              training_or_global_configuration_changed=False)
    raise RuntimeError('No read-only observation route returned')
