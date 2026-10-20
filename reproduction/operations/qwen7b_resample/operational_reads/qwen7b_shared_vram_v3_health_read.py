import subprocess,json,os,time
unit='sst-q7b-g5-sharedvram-v3-70188b8e50fd.service'
show=subprocess.check_output(['systemctl','show',unit,'-p','MainPID','-p','ActiveState','-p','ControlGroup'],text=True)
v=dict(epoch=time.time(),unit_show=show)
p=dict(x.split('=',1)for x in show.splitlines()if'='in x)
if p.get('MainPID')=='1259750'and p.get('ActiveState')=='active':
 r=subprocess.run(['nsenter','-t','1259750','-n','/usr/bin/python3','-c','import urllib.request; print(urllib.request.urlopen("http://127.0.0.1:28831/health",timeout=10).read().decode())'],env=dict(os.environ,PYTHONUTF8='1'),text=True,capture_output=True,timeout=20);v['health']=dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
print(json.dumps(v))
