import json,subprocess,time
v={"epoch":time.time()}
for name,args in [("node",["scontrol","show","node","h100-5","--oneliner"]),("holder",["scontrol","show","job","3349","--oneliner"]),("memory",["cat","/proc/meminfo"])]:
 r=subprocess.run(args,text=True,capture_output=True,timeout=15);v[name]={"rc":r.returncode,"stdout":r.stdout,"stderr":r.stderr}
print(json.dumps(v))
