"""Prepare an immutable paired-checkpoint inventory on8; start one copy on10."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from reproduction.h1008_qprobe.management_transport import remote

PREPARE='''import pathlib,json,hashlib,importlib.util,os
P=pathlib.Path;meta=META;base=P(meta['base']);run=json.loads((base/'logs/run-manifest.json').read_text())
assert run['run_id']==meta['run_id'] and run['question_probe']['loss_version']in ('std_class_ce_v1','binary_mix_variance_v1')
step=meta['step'];receipt=base/'artifacts/question-probe/completed-pairs'/('global_step_'+str(step))/'source-receipt.json'
root=base/'preparation'/('independent-backup-global_step_'+str(step)+'-once-v1')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def once(p,value):
 with p.open('x') as f:json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())
 fd=os.open(p.parent,os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)
if root.exists():
 p=root/'RECEIPT.json'
 if not p.exists():print(json.dumps(dict(status='EXISTING_INTENT_INSPECT_REQUIRED',root=str(root))));raise SystemExit
 previous=json.loads(p.read_text());assert previous['run_id']==meta['run_id'] and previous['step']==step and sha(P(previous['inventory_path']))==previous['inventory_sha256']
 print(json.dumps(dict(previous,action='existing immutable inventory verified; no replay')));raise SystemExit
if not receipt.exists():print(json.dumps(dict(status='PENDING_COMPLETE_PAIRED_CHECKPOINT',step=step,intent_created=False,expected_receipt=str(receipt))));raise SystemExit
pair=json.loads(receipt.read_text());assert pair['run_id']==meta['run_id'] and pair['step']==step and pair['status']=='QPROBE_COMPLETED_PAIR_FULL_SHA_CPU_STATES_PASS'
tools=base/'preparation/backup-tools'/meta['tools_commit'];code=tools/'inventory.py';assert sha(code)==meta['inventory_code_sha256']
spec=importlib.util.spec_from_file_location('published_checkpoint_inventory',code);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
plan=module.inventory(receipt,sha(receipt),official_stage_root=meta['official_stage_root'])
for path in (code,tools/'INTENT.json',tools/'RECEIPT.json'):
 plan['files'].append(dict(source_path=str(path.resolve()),payload_path=str(path.resolve()).lstrip('/'),bytes=path.stat().st_size,expected_sha256=sha(path),roles=['published_backup_inventory_control']))
plan['files']=sorted(plan['files'],key=lambda row:row['source_path']);plan['bytes']=sum(row['bytes'] for row in plan['files']);plan['backup_control_commit']=meta['control_commit']
root.mkdir(mode=0o700);once(root/'INTENT.json',dict(run_id=meta['run_id'],step=step,paired_receipt_sha256=sha(receipt),control_commit=meta['control_commit'],copy_launched=False))
path=root/'inventory.json';once(path,plan)
result=dict(status='SOURCE_INVENTORY_READY',run_id=meta['run_id'],step=step,inventory_path=str(path),inventory_sha256=sha(path),files=len(plan['files']),bytes=plan['bytes'],copy_launched=False,semantic_reload_performed=False)
once(root/'RECEIPT.json',result);print(json.dumps(result))
'''

START='''import pathlib,json,hashlib,importlib.util,socket
P=pathlib.Path;meta=META;assert socket.gethostname().split('.')[0]=='h100-10'
base=P(meta['base']);source=P('/mnt/shared/workspace/training_data_node_8')/base.relative_to('/mnt/data');step=meta['step']
prepared=source/'preparation'/('independent-backup-global_step_'+str(step)+'-once-v1')/'RECEIPT.json'
if not prepared.exists():print(json.dumps(dict(status='PENDING_SOURCE_INVENTORY',step=step,copy_launched=False)));raise SystemExit
receipt=json.loads(prepared.read_text());assert receipt['status']=='SOURCE_INVENTORY_READY' and receipt['run_id']==meta['run_id'] and receipt['step']==step
inventory=P('/mnt/shared/workspace/training_data_node_8')/P(receipt['inventory_path']).relative_to('/mnt/data')
assert hashlib.sha256(inventory.read_bytes()).hexdigest()==receipt['inventory_sha256']
code=P('/mnt/data/shaoshitong/search-g1-qprobe3-independent-backup-20261015/published-tools')/meta['tools_commit']/'copy-worker.py'
assert hashlib.sha256(code.read_bytes()).hexdigest()==meta['copy_code_sha256']
spec=importlib.util.spec_from_file_location('published_independent_copy',code);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
root=P('/mnt/data/shaoshitong/search-g1-qprobe3-independent-backup-20261015')/(meta['run_id']+'-step'+str(step)+'-once-v1')
result=module.launch(root,inventory,receipt['inventory_sha256'],meta['run_id'])
print(json.dumps(dict(root=str(root),launch_result=result,inventory_sha256=receipt['inventory_sha256'],step=step)))
'''


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','start']);p.add_argument('--proof',type=Path,required=True);p.add_argument('--base',required=True);p.add_argument('--run-id',required=True);p.add_argument('--step',type=int,choices=[5,50,100,150,200],required=True);p.add_argument('--tools-commit',required=True);p.add_argument('--official-stage-root',required=True);p.add_argument('--control-path',required=True);p.add_argument('--evidence',required=True);a=p.parse_args()
    repo=Path(__file__).resolve().parents[2];commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo).decode().strip();proof=json.loads(a.proof.read_text())
    assert proof['published_control_commit']==commit and proof['all_blobs_full_sha_match']
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo)
    meta=dict(base=a.base,run_id=a.run_id,step=a.step,tools_commit=a.tools_commit,official_stage_root=a.official_stage_root,control_commit=commit,inventory_code_sha256=proof['files']['reproduction/h1008_qprobe/backup_inventory.py'],copy_code_sha256=proof['files']['reproduction/h1008_qprobe/independent_copy.py'])
    target='h100-8' if a.phase=='prepare' else 'h100-10';script=PREPARE if a.phase=='prepare' else START
    print(remote(script.replace('META',repr(meta)),a.evidence,a.control_path,target=target))


if __name__=='__main__':main()
