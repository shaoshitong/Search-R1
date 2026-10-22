import pathlib,json,socket,time
a=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021');o=a/'online-mixed-pilot-CPU-acceptance-20261022-v1';suite=a/'online-mixed-layer-prefix-20261022-2gpu-v1';rows=[]
for name in('layer27-prefix20','layer21-prefix20'):
 v=json.loads((o/name/'accepted.json').read_bytes());p=pathlib.Path(v['checkpoint']['path']);s=p.stat();actual=dict(device=s.st_dev,inode=s.st_ino,bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns);prior=v['file_stat'];rows.append(dict(variant=name,prior=prior,current=actual,differences={k:[prior[k],actual[k]]for k in prior if prior[k]!=actual[k]},latest_matches_accepted=json.loads((suite/name/'latest.json').read_bytes())==v['checkpoint']))
print(json.dumps(dict(host=socket.gethostname(),time=time.time(),rows=rows)))
