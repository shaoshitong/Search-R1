"""Task-local remote X11 control; actions only in stdin, screenshots in memory."""
import base64
import json
import subprocess
import sys

actions = json.load(sys.stdin)
remote = '''import os,subprocess,base64,time,json
env=dict(os.environ,DISPLAY=":197",XAUTHORITY="/run/h3c-inode-task/Xauthority")
for action in ACTIONS:
    if action[0]=="text":
        subprocess.run(["xdotool","type","--clearmodifiers","--file","-"],input=action[1],text=True,env=env,check=True,capture_output=True)
    else:
        subprocess.run(["xdotool"]+action,env=env,check=True,capture_output=True)
time.sleep(.3)
p=subprocess.run(["import","-window","root","png:-"],env=env,capture_output=True,check=True)
print("data:image/png;base64,"+base64.b64encode(p.stdout).decode())
'''.replace("ACTIONS", repr(actions))
p = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                    "mgmt-host", "sudo -n python3 -"], input=remote,
                   capture_output=True, text=True, timeout=25)
if p.returncode:
    # Do not print the remote script or input actions, which can contain a secret.
    print("Remote UI action failed; return code", p.returncode)
    sys.exit(1)
print(p.stdout.strip())
