import subprocess,json,time
out={'time':time.time(),'read_only':True}
for name in('acd_u','acd_ue'):
 p=subprocess.run(['scontrol','show','partition',name,'-o'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=20);out[name]={'returncode':p.returncode,'stdout':p.stdout.strip(),'stderr':p.stderr[-1000:]}
print(json.dumps(out))
