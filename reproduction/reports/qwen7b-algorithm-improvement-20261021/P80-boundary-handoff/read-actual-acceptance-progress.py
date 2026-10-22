import pathlib,json,time
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-mixed-P80-handoff-CPU-acceptance-20261023-v1')
files=['controller.json','progress.json','failure.json','layer18-prefix80/epoch-01/accepted.json','layer15-prefix80/epoch-01/accepted.json']
print(json.dumps(dict(time=time.time(),files={n:(json.loads((b/n).read_bytes())if(b/n).is_file()else None)for n in files})))
