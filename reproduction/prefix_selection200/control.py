"""Code-only deployment and one native submission for prefix-selection200."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import time

from reproduction.prefix20_group_joint100.control import archive_for_commit


def import_resume_script(base):
    return r'''
import hashlib,json,os,pathlib,sys,time
b=pathlib.Path(BASE);c=json.loads((b/'preparation/launch-config.json').read_text());spec=c['resume_import'];source=pathlib.Path(spec['source_base'])
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def once(p,value):
 with p.open('x')as f:json.dump(value,f,sort_keys=True);f.flush();os.fsync(f.fileno())
prep=b/'preparation';receipt=prep/'resume-import-receipt.json';intent=prep/'resume-import-intent.json'
if receipt.exists():print(json.dumps(dict(status='EXISTING_RESUME_IMPORT_NO_REPLAY',receipt=read(receipt))));raise SystemExit
if intent.exists():print(json.dumps(dict(status='RESUME_IMPORT_INTENT_RECONCILE_REQUIRED')));raise SystemExit
assert b!=source and b.parent==source.parent and b.name.startswith('search-g1-prefix-selection200-') and source.resolve()==source
terminal=pathlib.Path(spec['terminal_proof']);assert sha(terminal)==spec['terminal_proof_sha256'];t=read(terminal)
assert not t['actual_births_present'] and not t['native_job_cgroup_paths'] and not t['gpu']['stdout'].strip()
original=read(source/'preparation/launch-config.json');assert sha(source/'preparation/launch-config.json')==spec['source_config_sha256']
assert original['run_id']==c['run_id'] and not(source/'actor/latest.json').exists()
assert read(source/'probe/latest.json')['completed_outer']==0
science=('model_revision','model_files_sha256','candidate_questions','selected_questions','responses_per_question','prefix_policy_tokens','exploration_questions','probe_learning_rate','probe_epochs_per_outer','probe_updates_per_outer','probe_global_batch_average','actor_global_mini_batch','actor_ppo_epochs','seed','official_model_dir','encoder_model_dir','canonical_train_sha256','fixed_test_manifest_sha256','grounding_probe_version')
assert all(c[key]==original[key] for key in science)
for phase,digest in spec['phase_hashes'].items():assert sha(source/'outers/outer-000001'/phase/'complete.json')==digest
prefix=read(source/'outers/outer-000001/prefix/complete.json');score=read(source/'outers/outer-000001/probe-score/manifest.json')
assert prefix['actor_version']==score['actor_version']==c['run_id']+'-official0'
assert prefix['trajectories']==25600 and score['count']==2560
verified=[]
def verify(p,producer):
 assert p.stat().st_size==producer['bytes'] and sha(p)==producer['sha256']
 verified.append(dict(path=str(p),bytes=producer['bytes'],sha256=producer['sha256']))
for row in prefix['ranks']:
 for key in ('states','prefix'):verify(pathlib.Path(row[key]['path']),row[key])
scan=source/'outers/outer-000001/prefix/scan'
for name,producer in read(scan/'scan-complete.json')['files'].items():verify(scan/name,producer)
for producer in score['shards']:verify(source/'outers/outer-000001/probe-score'/producer['file'],producer)
verify(source/'probe/latest.pt',read(source/'probe/latest.json'))
once(intent,dict(run_id=c['run_id'],source_base=str(source),created_unix=time.time(),terminal_proof_sha256=spec['terminal_proof_sha256']))
assets=[]
def link_file(s,d):
 assert s.is_file() and not s.is_symlink() and s.resolve()==s
 d.parent.mkdir(parents=True,exist_ok=True);assert not d.exists();os.link(s,d)
 info=s.stat();assets.append(dict(source=str(s),target=str(d),bytes=info.st_size,inode=info.st_ino))
def link_tree(s,d):
 for p in sorted(s.rglob('*')):
  assert not p.is_symlink()
  if p.is_file():link_file(p,d/p.relative_to(s))
link_tree(source/'probe',b/'probe')
for name in ('binding.json','state.json'):link_file(source/'question-pool'/name,b/'question-pool'/name)
src=source/'outers/outer-000001';dst=b/'outers/outer-000001'
for name in ('pool-reservation.json','selection.json','questions.json','actor-binding.json'):link_file(src/name,dst/name)
for phase in ('prefix','probe-score'):link_tree(src/phase,dst/phase)
if 'full' in spec['phase_hashes']:
 link_tree(src/'full',dst/'full')
# A source rank receipt is reused only if it was actually complete; incomplete
# gzip/native logs stay at source. New workers continue those same original G5s.
for rank,digest in c['continue_reuse']['imported_receipts_sha256'].items():
 p=src/'full'/('rank-%s-complete.json'%rank);assert sha(p)==digest
 if not(dst/'full'/p.name).exists():link_file(p,dst/'full'/p.name)
value=dict(status='EXACT_COMPLETE_PREFIX_SCORE_AND_PROBE0_IMPORTED',run_id=c['run_id'],outer=1,source_base=str(source),phase_hashes=spec['phase_hashes'],source_config_sha256=spec['source_config_sha256'],imported_full_rank_receipts=c['continue_reuse']['imported_receipts_sha256'],assets=assets,independent_active_asset_full_sha_readback=verified,model_or_data_via_Mac=False,head_reset=False,new_prefixes_generated=0,source_originals_modified=False)
once(receipt,value);print(json.dumps(dict(status=value['status'],run_id=c['run_id'],outer=1,phase_hashes=value['phase_hashes'],imported_full_rank_receipts=value['imported_full_rank_receipts'],assets_count=len(assets),source_originals_modified=False)))
'''.replace('BASE', repr(base))


def deployment_script(base, commit, config, built, reuse_commit=None, reuse_repo=None):
    payload = dict(base=base, commit=commit, config=dict(config, base=base, source_commit=commit),
        files=built['files'], modes=built['modes'], archive=base64.b64encode(built['archive']).decode(),
        reuse_files=built['reuse_files'], reuse_commit=reuse_commit, reuse_repo=reuse_repo)
    return r'''
import base64,hashlib,io,json,os,pathlib,shutil,sys,tarfile,time
p=PAYLOAD;b=pathlib.Path(p['base']);c=p['config']
def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def once(path,value):
 with path.open('x')as handle:json.dump(value,handle,sort_keys=True,allow_nan=False);handle.flush();os.fsync(handle.fileno())
assert str(b).startswith('/mnt/data/shaoshitong/search-g1-prefix-selection200-')and'..'not in b.parts
if b.exists():
 print(json.dumps(dict(status='EXISTING_DEPLOY_NAMESPACE_NO_REPLAY',intent=(b/'preparation/deploy-intent.json').exists(),receipt=read(b/'preparation/deploy-receipt.json')if(b/'preparation/deploy-receipt.json').exists()else None)));raise SystemExit
# Reuse already independently validated source weights. Read the small actual
# CPU proof plus exact source stat; do not repeat an extra GPU forward gate.
proof_path=pathlib.Path(c['probe_source_cpu_proof']);assert sha(proof_path)==c['probe_source_cpu_proof_sha256'];proof=read(proof_path)
source=pathlib.Path(c['probe_source_checkpoint']);stat=source.stat()
assert stat.st_size==c['probe_source_bytes'] and stat.st_ino==c['probe_source_inode']
assert c['probe_source_sha256'] in json.dumps(proof)
assert sha(pathlib.Path(c['fixed_test_features'])/'manifest.json')==c['fixed_test_manifest_sha256']
assert sha(pathlib.Path(c['canonical_train']))==c['canonical_train_sha256']
assert sha(pathlib.Path(c['grounding_manifest']))==c['grounding_manifest_sha256']
for key in ('validation_questions','final_questions'):
 assert sha(pathlib.Path(c[key]))==c[key+'_sha256']
b.mkdir(mode=0o700);prep=b/'preparation';prep.mkdir();(b/'repo').mkdir();(b/'logs').mkdir()
once(prep/'deploy-intent.json',dict(run_id=c['run_id'],source_commit=p['commit'],file_count=len(p['files']),reuse_commit=p['reuse_commit'],created_unix=time.time()))
if p['reuse_files']:
 root=pathlib.Path(p['reuse_repo']);assert root.resolve(strict=True)==root and root.name=='repo'
 for name,digest in p['reuse_files'].items():
  rel=pathlib.PurePosixPath(name);assert not rel.is_absolute()and'..'not in rel.parts
  source=root/name;assert source.is_file()and not source.is_symlink()and sha(source)==digest,name
  target=b/'repo'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target);target.chmod(p['modes'][name])
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(p['archive'])))as handle:
 for member in handle:
  rel=pathlib.PurePosixPath(member.name);assert not rel.is_absolute()and'..'not in rel.parts
  if member.isdir():(b/'repo'/member.name).mkdir(parents=True,exist_ok=True)
  elif member.isfile():
   target=b/'repo'/member.name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(handle.extractfile(member).read());target.chmod(member.mode)
  else:raise ValueError('Code archive must contain regular files only')
for name,digest in p['files'].items():assert sha(b/'repo'/name)==digest,name
sys.path.insert(0,str(b/'repo'))
from reproduction.prefix_selection200.launch import validate_config
validate_config(c,b)
once(prep/'launch-config.json',c);once(prep/'source-files.json',p['files'])
result=dict(status='PREFIX_SELECTION200_CODE_DEPLOYED_NOT_SUBMITTED',base=str(b),run_id=c['run_id'],source_commit=p['commit'],target_code_files=len(p['files']),all_target_code_full_sha_match=True,config_sha256=sha(prep/'launch-config.json'),reused_code_files=len(p['reuse_files']),code_archive_bytes=len(base64.b64decode(p['archive'])),large_payload_transferred_from_Mac=False)
once(prep/'deploy-receipt.json',result);print(json.dumps(result))
'''.replace('PAYLOAD', repr(payload))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('deploy', 'import-resume', 'import-joint', 'import-lr50', 'submit', 'query', 'observe'))
    parser.add_argument('--base', required=True); parser.add_argument('--commit'); parser.add_argument('--config')
    parser.add_argument('--receipts', required=True); parser.add_argument('--reuse-source-commit'); parser.add_argument('--reuse-source-repo')
    args = parser.parse_args()
    if bool(args.reuse_source_commit) != bool(args.reuse_source_repo):
        parser.error('Both source reuse fields are required together')
    if args.phase == 'deploy':
        if not args.commit or not args.config: parser.error('Deploy requires a published commit/config')
        built = archive_for_commit(args.commit, args.reuse_source_commit)
        code = deployment_script(args.base, args.commit, json.loads(Path(args.config).read_text()), built,
                                 args.reuse_source_commit, args.reuse_source_repo)
    elif args.phase == 'import-lr50':
        code = ('import sys,json;sys.path.insert(0,'+repr(args.base+'/repo')+');'
                'from reproduction.prefix_selection200.resume_probe_lr50 import import_complete50;'
                'print(json.dumps(import_complete50('+repr(args.base)+'),allow_nan=False))')
    elif args.phase == 'import-joint':
        code = ('import sys,json;sys.path.insert(0,'+repr(args.base+'/repo')+');'
                'from reproduction.prefix_selection200.resume import import_complete;'
                'print(json.dumps(import_complete('+repr(args.base)+'),allow_nan=False))')
    elif args.phase == 'import-resume':
        code = import_resume_script(args.base)
    else:
        code = 'import sys,json;sys.path.insert(0,'+repr(args.base+'/repo')+');from reproduction.prefix_selection200.launch import '+args.phase+';print(json.dumps('+args.phase+'('+repr(args.base)+'),allow_nan=False))'
    from reproduction.h1008_qprobe.management_transport import remote
    path = Path(args.receipts) / ('prefix-selection200-'+args.phase+'-'+str(time.time_ns())+'.json')
    result = json.loads(remote(code, path, control_path=None, target='h100-8'))
    output = path.with_name(path.stem+'-result.json'); output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(path=str(output), sha256=hashlib.sha256(output.read_bytes()).hexdigest(), result=result), allow_nan=False))


if __name__ == '__main__': main()
