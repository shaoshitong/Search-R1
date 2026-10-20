import pathlib,os,json,time
p=pathlib.Path('/data/user/jxu957/experiments/dynrpo-prefix5-thread5-20261017-r4/actor/outers/outer-0200-attempt-d3eadfec8ccd440f82136d69c35cf237/huggingface');assert os.getuid()==21930
files=sorted(p.glob('*.safetensors'));assert len(files)==4
changed=[]
for f in files:
 s=f.stat();assert s.st_uid==os.getuid() and s.st_gid==21919 and s.st_mode&0o777==0o600
 os.chmod(f,0o640);n=f.stat();assert n.st_size==s.st_size and n.st_mtime_ns==s.st_mtime_ns
 changed.append(dict(path=str(f),old_mode='0600',new_mode='0640',shared_group=21919,bytes=n.st_size,tensor_bytes_or_mtime_modified=False))
print(json.dumps(dict(status='EXACT_FOUR_SHARDS_SHARED_GROUP_READ_GRANTED',changed=changed,checked_unix=time.time())))
