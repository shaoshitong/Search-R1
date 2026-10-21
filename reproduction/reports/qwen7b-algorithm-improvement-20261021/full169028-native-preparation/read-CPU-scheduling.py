"""Read the actual allowed CPU scheduling choices; no mutations."""
import json
import subprocess
import time
commands = {
    'associations': ['sacctmgr', '-n', '-P', 'show', 'assoc', 'where', 'user=mou176', 'format=Cluster,Account,User,Partition,QOS,DefaultQOS'],
    'qos': ['sacctmgr', '-n', '-P', 'show', 'qos', 'format=Name,Priority,MaxWall,MaxTRESPerUser'],
    'debug': ['scontrol', 'show', 'partition', 'debug', '-o'],
    'debug_nodes': ['sinfo', '-h', '-N', '-p', 'debug', '-o', '%N|%T|%C|%m|%G'],
    'partitions': ['sinfo', '-h', '-o', '%P|%a|%l|%D|%G'],
}
out = dict(epoch=time.time())
for name, command in commands.items():
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=15)
    out[name] = dict(returncode=result.returncode, stdout=result.stdout[:16000], stderr=result.stderr[:1000])
print(json.dumps(out, sort_keys=True))
