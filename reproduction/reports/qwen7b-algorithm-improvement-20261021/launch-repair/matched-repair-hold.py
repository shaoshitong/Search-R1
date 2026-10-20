import subprocess,json,time
out={}
for j,v in [('742839','joint'),('742840','event')]:
 s=subprocess.check_output(['scontrol','show','job',j,'-o'],universal_newlines=True,timeout=15);assert 'JobName=q7b-loss128-'+v+' ' in s and 'UserId=mou176(' in s
 if 'JobState=PENDING ' in s:subprocess.check_call(['scontrol','hold',j],timeout=15)
 out[j]=subprocess.check_output(['scontrol','show','job',j,'-o'],universal_newlines=True,timeout=15)
print(json.dumps(dict(time=time.time(),jobs=out)))
