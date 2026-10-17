"""Deploy exact a61b481 source once, then perform a separate full-SHA HK readback."""
import base64,gzip,hashlib,importlib.util,json,subprocess,sys,time
from pathlib import Path
WS=Path("/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6")
W=WS/'work/easy2560-co-rollout-hpc3-4gpu-20261017'
OUT=WS/'outputs/easy2560-co-rollout-hpc3-4gpu-20261017'
COMMIT='a61b48139386b346e4fef12511ece1ac0d09a8e3'
TAG='recovery5-after730961-a61b-HK'
sys.path.insert(0,str(W))
from reproduction.easy2560_joint4.deploy_source import published_payload
entry=W/'reproduction/easy2560_joint4/code_backup/recovery5-after730912-hk-source-8033-20261018/deploy_exact8033_HK_source_once.py'
spec=importlib.util.spec_from_file_location('existing_single_transport',entry)
transport=importlib.util.module_from_spec(spec);spec.loader.exec_module(transport)
payload=published_payload(W,COMMIT,"/data/user/jxu957/experiments/easy2560-corollout-joint4-recovery5-after730961-20261018")
operator=subprocess.check_output(['git','show',COMMIT+':'+transport.OPERATOR],cwd=W)
assert hashlib.sha256(operator).hexdigest()==transport.OPERATOR_SHA
transport.save(OUT/(TAG+'-single-attempt-start.json'),dict(source_commit=COMMIT,created_unix=time.time(),automatic_retry=False,jobs_submitted=0,controller_started=False))
deployed=transport.run_ssh(operator.decode(),dict(commit=COMMIT,manifest=payload['manifest'],tar_gz=payload['tar_gz']),OUT/(TAG+'-deployment.json'))
verified=transport.run_ssh(transport.VERIFY,dict(commit=COMMIT,manifest=payload['manifest'],publication_sha256=deployed['publication_sha256']),OUT/(TAG+'-independent-readback.json'))
assert verified['files']==len(payload['manifest']) and verified['bytes']==sum(r['bytes'] for r in payload['manifest'].values())
result=dict(status='PUBLISHED_SOURCE_AND_SECOND_FULL_SHA_PASS',source_commit=COMMIT,files=verified['files'],bytes=verified['bytes'],publication_sha256=verified['publication_sha256'],source_root=verified['source_root'],checked_unix=time.time(),jobs_submitted=0,controller_started=False,automatic_retry=False)
transport.save(OUT/(TAG+'-completion.json'),result)
print(json.dumps(result))

