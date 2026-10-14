import pathlib,subprocess,hashlib,json,datetime
repo=pathlib.Path('/Users/shaoshitong/Documents/Search-G1/qprobe-h1008-transformer3');bare=pathlib.Path('work/search-g1-mixed-publication-readback-584db0b.git').resolve()
def git(args,path):return subprocess.check_output(['git']+args,cwd=path)
commit=git(['rev-parse','HEAD'],repo).decode().strip();assert commit.startswith('86c80e96')
subprocess.run(['git','fetch','git@github.com:shaoshitong/Search-R1.git','codex/qprobe-h1008-transformer3-20261015'],cwd=bare,check=True,capture_output=True)
assert git(['rev-parse','FETCH_HEAD'],bare).decode().strip()==commit
files={}
for row in git(['ls-tree','-rz',commit],repo).split(b'\0'):
 if not row:continue
 meta,name=row.split(b'\t',1);mode,kind,blob=meta.split();assert kind==b'blob'
 a=git(['cat-file','blob',blob.decode()],repo);b=git(['cat-file','blob',blob.decode()],bare)
 sha=hashlib.sha256(a).hexdigest();assert sha==hashlib.sha256(b).hexdigest();files[name.decode()]=sha
result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),published_control_commit=commit,previous_experiment_frozen_training_source='0553adec146537e07b84377ce31cba41a54f37cd',new_experiment_started=False,verified_file_count=len(files),all_blobs_full_sha_match=True,files=files)
pathlib.Path('work/search-g1-publication-qprobe3-86c80e96-independent-readback.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='files'}))
