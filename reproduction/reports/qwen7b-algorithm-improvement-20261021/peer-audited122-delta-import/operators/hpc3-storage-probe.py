import os,pathlib,subprocess,json,socket
p=pathlib.Path("/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/peer-newseed-h100-20261021/audited122-delta-20261021-673b8cb6");p.mkdir(parents=True,exist_ok=True)
probe=p/('.audited122-rw-probe-'+str(os.getpid()));fd=os.open(str(probe),os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.write(fd,b'raw-delta-staging-write-probe');os.fsync(fd);os.close(fd);probe.unlink()
s=os.statvfs(str(p));free=s.f_bavail*s.f_frsize;assert free>3*1024**3,'Insufficient independent staging space'
print(json.dumps(dict(host=socket.gethostname(),path=str(p),free_bytes=free,write_probe_fsync_removed=True,df=subprocess.check_output(['df','-B1',str(p)],universal_newlines=True),mount=subprocess.check_output(['findmnt','-T',str(p),'-o','TARGET,SOURCE,FSTYPE,OPTIONS','-n'],universal_newlines=True))))
