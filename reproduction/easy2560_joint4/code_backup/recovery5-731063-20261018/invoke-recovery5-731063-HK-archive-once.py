import importlib.util,json,re,subprocess,time
from pathlib import Path
w=Path("/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6/work/easy2560-co-rollout-hpc3-4gpu-20261017");out=Path("/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6/outputs/easy2560-co-rollout-hpc3-4gpu-20261017")
oldroot=w/'reproduction/easy2560_joint4/code_backup/recovery5-730961-20261018'
oldbase="/data/user/jxu957/experiments/easy2560-corollout-joint4-recovery5-after730912-20261018";newbase="/data/user/jxu957/experiments/easy2560-corollout-joint4-recovery5-after730961-20261018"
commit="a61b48139386b346e4fef12511ece1ac0d09a8e3"
files={}
for kind in ('launch','observe'):
 name=kind+'-recovery5-730961-HK-archive-controller.py'
 raw=(oldroot/name).read_text().replace('730961','731063').replace(oldbase,newbase).replace('8033adb58beb85bbf9ec5815486ca710a77d3177',commit)
 target=out/(kind+'-recovery5-731063-HK-archive-controller.py');target.write_text(raw);files[kind]=target
spec=importlib.util.spec_from_file_location('existing_single_transport',w/'reproduction/easy2560_joint4/code_backup/recovery5-after730912-hk-source-8033-20261018/deploy_exact8033_HK_source_once.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
secret=Path('/Users/shaoshitong/Documents/remote_codex_control/memories/hpc3-credentials.private.md').read_text()
line=next(x for x in secret.splitlines() if x.startswith('账号 jxu957'))
password=re.split(r'密码\s*[:：]?\s*',line,1)[1].strip().strip('`');secret=line=None
payload=dict(password=password,job='731063',base=newbase,source_commit=commit,run_id='dfe9c94084bb48b2a9bb7b60f9d882da',source_root='/backup/shaoshitong/easy2560-corollout-hpc3-4gpu-20261017/source/'+commit)
result=m.run_ssh(files['launch'].read_text(),payload,out/'recovery5-731063-HK-archive-launch-once.json')
password=payload=None
print(json.dumps(result))

