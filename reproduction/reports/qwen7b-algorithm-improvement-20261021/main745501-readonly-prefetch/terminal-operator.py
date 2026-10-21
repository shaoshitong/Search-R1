import pathlib,json,hashlib,subprocess,time
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/joint124194-reader-prefetch-v1');main=b.parent/'joint-balanced-native-E4-124194-v1'
def fp(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
pub=json.loads((b/'source-publication.json').read_text());done=json.loads((b/'complete.json').read_text());assert pub['commit']=='d21968bd9249d85d9043d38a1d04da10b281407f'and done['status']=='FINITE_READ_ONLY_PREFETCH_FINISHED'and done['full_file_SHA_verified']and done['main_checksums_unchanged']and done['last_main_manifest_index']==3951
for name,item in pub['files'].items():assert fp(b/name)==item
assert fp(main/'config.json')['sha256']=='5aaa63c9348f410357be92ce491ba6d18286d618ab86dae6e63f193f2b4367b6';assert json.loads((main/'source-publication.json').read_text())['commit']=='3b56e583efe2469ac9108873c31de7144a5201e6'
r=subprocess.run(['sacct','-n','-P','-j','745501','--format=JobID,JobName,State,ExitCode,Elapsed'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=12);assert r.returncode==0
out=dict(status='FINITE_PREFETCH_COMPLETION_AND_UNCHANGED_MAIN_BINDING_FRESHLY_VERIFIED',time=time.time(),files={name:fp(b/name)for name in('source-publication.json','worker.py','config.json','launch.json','process.json','progress.json','complete.json','step.log')},summary=done,main_config=fp(main/'config.json'),main_publication=fp(main/'source-publication.json'),sacct=r.stdout.strip(),no_model_tensors_or_final_payload_read=True)
print(json.dumps(out))
