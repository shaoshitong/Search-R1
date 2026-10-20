from pathlib import Path
import subprocess,json,time
v=dict(epoch=time.time(),files={})
for name in ['/etc/slurm/cgroup.conf','/etc/slurm-llnl/cgroup.conf']:
 p=Path(name)
 if p.is_file():v['files'][name]=[x for x in p.read_text().splitlines()if x.strip()and not x.lstrip().startswith('#')and any(k.lower()in x.lower()for k in ['ConstrainDevices','CgroupPlugin','AllowedDevicesFile','ConstrainRAMSpace'])]
r=subprocess.run(['scontrol','show','config'],text=True,capture_output=True,timeout=15);v['slurm']=[x.strip()for x in r.stdout.splitlines()if x.strip().startswith(('TaskPlugin ','TaskPluginParam ','PrologFlags ','SelectType ','SelectTypeParameters '))];print(json.dumps(v))
