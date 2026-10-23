import subprocess,json,time
from pathlib import Path
jobs="750396,750432,750029,745198,750471"
rows={}
for key,cmd in [("queue",["squeue","-h","-j",jobs,"-o","%i|%u|%T|%N"]),("accounting",["sacct","-n","-X","-j",jobs,"-P","--format=JobIDRaw,User,State,ExitCode,NodeList"])]:
 p=subprocess.run(cmd,capture_output=True,text=True,timeout=20);rows[key]={"returncode":p.returncode,"stdout":p.stdout,"stderr":p.stderr}
print(json.dumps({"epoch":time.time(),"slurm":rows}))
