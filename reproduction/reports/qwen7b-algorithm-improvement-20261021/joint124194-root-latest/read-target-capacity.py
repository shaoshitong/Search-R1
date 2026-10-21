import pathlib,shutil,json,time,subprocess
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019');d=shutil.disk_usage(str(b));r=subprocess.run(['df','-PT',str(b)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=12)
print(json.dumps(dict(epoch=time.time(),path=str(b),filesystem_capacity=dict(total=d.total,used=d.used,free=d.free),df=r.stdout.strip(),per_user_quota_not_proven_by_df=True,model_tensor_loaded=False,modified=False)))
