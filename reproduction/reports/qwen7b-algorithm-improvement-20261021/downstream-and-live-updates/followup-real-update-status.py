import json,pathlib,time
root=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/matched-loss128-followup-v1/event')
out={}
for name in ('within1','inter1','bothrank1','mean1','var5','mean1-var5','bothrank1-mean1-var5','inter1-hard16'):
    base=root/name
    own={}
    for filename in ('controller.json','next-update-replay.json','progress.json','latest.json','failure.json','complete.json'):
        path=base/filename
        if path.exists():
            value=json.loads(path.read_text())
            if filename=='controller.json':
                value={k:value.get(k) for k in ('pid','job_id','source_commit','epoch_time')}
            elif filename=='progress.json':
                value={k:value.get(k) for k in ('cursor','steps','budget','epoch_time','max_gpu_allocated')}
            elif filename=='latest.json':
                value={k:value.get(k) for k in ('path','bytes','sha256','cursor','steps','epoch_time')}
            elif filename=='complete.json':
                value={k:value.get(k) for k in ('status','cursor','steps','epoch_time','checkpoint')}
            own[filename]=value
    out[name]=own
print(json.dumps(dict(time=time.time(),variants=out)))
