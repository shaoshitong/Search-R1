"""Single-session retained50 LR recovery; every mutation has existing once receipts."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

from reproduction.prefix_selection200.control import deployment_script
from reproduction.prefix20_group_joint100.control import archive_for_commit


def build(config, commit, source_commit, source_repo):
    built = archive_for_commit(commit, source_commit)
    deployment = deployment_script(config['base'], commit, config, built, source_commit, source_repo)
    # Bind the execution configuration to the actual stopped-state proof read in
    # this same remote session, rather than inventing a pending proof SHA.
    deployment = deployment.replace(
        "p=PAYLOAD;b=pathlib.Path(p['base']);c=p['config']",
        "p=PAYLOAD;b=pathlib.Path(p['base']);c=p['config']")
    marker = "p=" + repr(dict(base=config['base'],commit=commit,config=dict(config,base=config['base'],source_commit=commit),files=built['files'],modes=built['modes'],archive=__import__('base64').b64encode(built['archive']).decode(),reuse_files=built['reuse_files'],reuse_commit=source_commit,reuse_repo=source_repo))
    assert marker in deployment
    replacement = marker + ";p['config']['resume_joint']['terminal_proof_sha256']=TERMINAL_SHA"
    deployment = deployment.replace(marker, replacement)
    helper = (Path(__file__).with_name('human_stop_terminal.py')).read_text()
    payload = dict(config=config, commit=commit, deployment=deployment,
                   terminal_helper=helper, terminal_helper_sha=hashlib.sha256(helper.encode()).hexdigest())
    return r"""
import hashlib,json,os,pathlib,subprocess,sys
p=PAYLOAD;c=p['config'];b=pathlib.Path(c['base']);s=c['resume_joint'];source=pathlib.Path(s['source_base'])
def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
proof=pathlib.Path(s['terminal_proof'])
if not proof.exists():
 folder=source/'preparation/normal-stop-tools'/p['terminal_helper_sha'];folder.mkdir(parents=True,exist_ok=True)
 tool=folder/'human_stop_terminal.py'
 if tool.exists():assert sha(tool)==p['terminal_helper_sha']
 else:tool.write_text(p['terminal_helper'])
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(source/'repo'),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
 r=subprocess.run([str(pathlib.Path(c['environment_root'])/'bin/python'),str(tool),'--base',str(source),'--job',str(s['source_job_id']),'--stop-receipt',str(source/'preparation/human-resume50-lr1e5-stop-once-v1/RECEIPT.json')],env=env,text=True,capture_output=True,timeout=150)
 assert r.returncode==0,(r.stdout,r.stderr)
 if not proof.exists():print(json.dumps(dict(status='PENDING_ACTUAL_TERMINAL_RELEASE',result=r.stdout)));raise SystemExit
ended=read(proof)
assert ended['status']=='HUMAN_STOP_ORIGINAL_INODE_FULL_SHA_OWNED_RELEASED' and ended['job_id']==str(s['source_job_id']) and ended['run_id']==c['run_id']
assert not ended['actual_births_present'] and not ended['native_job_cgroup_paths'] and not ended['owned_gpu']
terminal_sha=sha(proof)
if not b.exists():
 code=p['deployment'].replace('TERMINAL_SHA',repr(terminal_sha))
 exec(compile(code,'<isolated_code_deployment>','exec'),{})
else:
 assert (b/'preparation/deploy-receipt.json').exists(),'Reconcile partial deployment intent, do not replay'
actual=read(b/'preparation/launch-config.json')
assert actual['source_commit']==p['commit'] and actual['resume_joint']['terminal_proof_sha256']==terminal_sha and actual['probe_learning_rate']==1e-5
repo=b/'repo';sys.path.insert(0,str(repo))
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(repo),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
r=subprocess.run([str(pathlib.Path(c['environment_root'])/'bin/python'),'-m','reproduction.prefix_selection200.resume_probe_lr50','--base',str(b)],env=env,cwd=str(repo),capture_output=True,text=True,timeout=180)
assert r.returncode==0,(r.stdout,r.stderr)
from reproduction.prefix_selection200.launch import submit
submitted=submit(b)
print(json.dumps(dict(status='LR50_RECOVERY_CHAIN_RESULT',base=str(b),source_commit=p['commit'],terminal_proof_sha256=terminal_sha,source_latest_before_stop=ended['latest_complete_outer'],deployment=read(b/'preparation/deploy-receipt.json'),import_receipt=read(b/'preparation/resume-joint-receipt.json'),submit=submitted)))
""".replace('PAYLOAD',repr(payload))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);parser.add_argument('--evidence',required=True)
    args=parser.parse_args();plan=json.loads(Path(args.plan).read_text());config=plan['config_draft']
    repo=Path(__file__).resolve().parents[2]
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=repo,text=True).strip()
    published=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+branch],cwd=repo,text=True).split()[0]
    assert commit==published
    source=config['resume_joint']['source_base']
    script=build(config,commit,'848a67a41a289aa0d9a04b907f852ef61d38cc63',source+'/repo')
    from reproduction.h1008_qprobe.established_controller_transport import remote
    output=remote(script,args.evidence)
    target=Path(args.evidence).with_suffix('.actual.json');target.write_text(output)
    value=json.loads(output.splitlines()[-1]);print(json.dumps(dict(actual=str(target),sha256=hashlib.sha256(target.read_bytes()).hexdigest(),result=value)))


if __name__=='__main__':main()
