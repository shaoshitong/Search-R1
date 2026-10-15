"""Plan immutable paired-state backups; a plan is never an independent SHA pass.

Run on the source host after a completed paired receipt exists. Copy/readback and
actual semantic restoration are separate operations, never implied here.
"""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def inventory(receipt_path, expected_receipt_sha, *, allowed_roots=('/mnt/data','/dataset')):
    receipt_path=Path(receipt_path).resolve()
    if digest(receipt_path)!=expected_receipt_sha:raise ValueError('Paired receipt SHA differs')
    pair=json.loads(receipt_path.read_text())
    if pair.get('status')!='QPROBE_COMPLETED_PAIR_FULL_SHA_CPU_STATES_PASS' or pair.get('schema_version')!=1:
        raise ValueError('Completed paired-state receipt required')
    step=pair['step'];base=Path(pair['source_base']).resolve();checkpoint=Path(pair['checkpoint']).resolve()
    if step not in (5,50,100,150,200) or pair['world_size']!=8 or pair['actor_snapshot_id']!=pair['run_id']+':'+str(step):
        raise ValueError('Unexpected checkpoint identity')
    if not checkpoint.is_relative_to(base):raise ValueError('Checkpoint outside source experiment')
    if sorted(row['rank'] for row in pair['boundary_worker_files'])!=list(range(8)):
        raise ValueError('All eight boundary RNG files required')
    rows={}
    def add(path, sha, role, size=None):
        original=Path(path);path=original.resolve()
        if not any(path.is_relative_to(Path(root).resolve()) for root in allowed_roots):raise ValueError('Not a remote artifact path')
        if not path.is_file():raise ValueError('Missing immutable artifact: '+str(path))
        if not isinstance(sha,str) or len(sha)!=64 or any(c not in '0123456789abcdef' for c in sha):raise ValueError('Malformed expected SHA')
        actual_size=path.stat().st_size
        if size is not None and actual_size!=size:raise ValueError('Receipt size changed')
        if str(path) in rows:
            if rows[str(path)]['expected_sha256']!=sha:raise ValueError('Conflicting dependency SHA')
            rows[str(path)]['roles'].append(role)
        else:rows[str(path)]=dict(source_path=str(path),payload_path=str(path).lstrip('/'),bytes=actual_size,expected_sha256=sha,roles=[role])
    def record(row,role):add(row['path'],row['full_read_sha256'],role,row['bytes'])
    add(receipt_path,expected_receipt_sha,'paired_receipt')
    for row in pair['files']:
        if not Path(row['path']).resolve().is_relative_to(checkpoint):raise ValueError('Checkpoint file escapes its root')
        record(row,'checkpoint')
    actual={str(p.resolve()) for p in checkpoint.rglob('*') if p.is_file()}
    if actual!={str(Path(row['path']).resolve()) for row in pair['files']}:
        raise ValueError('Paired checkpoint file set changed')
    for key in ('boundary_driver','source_run_manifest','reward_probe_manifest'):record(pair[key],key)
    for row in pair['boundary_worker_files']:record(row,'boundary_worker_rng')
    for row in pair['reward_files']:record(row,'reward_probe_or_encoder')
    run_path=Path(pair['source_run_manifest']['path'])
    if digest(run_path)!=pair['source_run_manifest']['full_read_sha256']:raise ValueError('Pinned run manifest changed')
    run=json.loads(run_path.read_text())
    if run['run_id']!=pair['run_id'] or run['training_world_size']!=8:raise ValueError('Run identity differs')
    probe=run['question_probe']
    if probe['probe_architecture']!='qwen2_decoder3_count6' or probe['exploration_questions']!=64 or probe['selected_questions']!=512:
        raise ValueError('Wrong question probe experiment')
    cache=Path(probe['cache_dir']).resolve();manifest=cache/'manifest.json'
    feature=json.loads(manifest.read_text())
    if feature['schema_version']!=2 or feature['spec_sha']!=pair['feature_spec_sha'] or feature['count']!=169028:
        raise ValueError('Frozen token feature identity differs')
    feature_sha=digest(manifest)
    launch=json.loads((base/'preparation/h1008-launch-config.json').read_text())
    reused=launch.get('precomputed_question_features')
    if reused and (Path(reused['path']).resolve()!=cache or reused['manifest_sha256']!=feature_sha):
        raise ValueError('Pinned reused feature manifest changed')
    add(manifest,feature_sha,'feature_manifest')
    add(base/'preparation/h1008-launch-config.json',digest(base/'preparation/h1008-launch-config.json'),'fresh_launch_config')
    for name,row in feature['files'].items():
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Feature file escapes cache')
        add(cache/relative,row['sha256'],'frozen_token_features',row['bytes'])
    for name,sha in run['initial_model_sha256'].items():add(Path(run['initial_model'])/name,sha,'official_actor_reference_initializer')
    # Root-disk /dataset is not exported to h100-10. Use only the completed,
    # byte-verified same-host staging aliases, retaining original payload paths.
    stage_root=base/'backup-dependencies/official-model-once-v1'
    stage_result=stage_root/'RESULT.json';aliases=[]
    if stage_result.exists():
        stage=json.loads(stage_result.read_text());intent=json.loads((stage_root/'INTENT.json').read_text())
        if stage.get('status')!='SAME_HOST_OFFICIAL_MODEL_STAGE_FULL_SHA_PASS' or stage['run_id']!=pair['run_id'] or intent['run_id']!=pair['run_id']:
            raise ValueError('Completed staging identity required')
        expected=run['initial_model_sha256'];original=Path(run['initial_model']).resolve()
        if intent['files_sha256']!=expected or Path(stage['source_model']).resolve()!=original:
            raise ValueError('Staged official source identity differs')
        staged_names=set()
        for item in stage['files']:
            source=Path(item['source_path']).resolve();target=Path(item['staged_path']).resolve();name=source.name
            if source!=original/name or name in staged_names or name not in expected or not target.is_relative_to(stage_root/'model'):
                raise ValueError('Staging alias escapes official dependency')
            if not target.is_file() or target.stat().st_size!=item['bytes']:
                raise ValueError('Staged dependency size changed')
            if any(item[key]!=expected[name] for key in ('expected_sha256','source_full_read_sha256','staged_full_read_sha256')):
                raise ValueError('Staging full SHA differs from official identity')
            rows[str(source)]['exported_source_path']=str(target)
            aliases.append(dict(original_source_path=str(source),exported_source_path=str(target),expected_sha256=expected[name]))
            staged_names.add(name)
        if staged_names!=set(expected):raise ValueError('Incomplete official staging aliases')
        for path in (stage_result,stage_root/'INTENT.json',stage_root/'LAUNCH.json'):add(path,digest(path),'staging_alias_evidence')
        code=[p for p in (base/'preparation').glob('stage-official-backup-*.py') if digest(p)==intent['worker_source_sha256']]
        if not code:raise ValueError('Published staging worker source missing')
        for path in code:add(path,intent['worker_source_sha256'],'staging_worker_source')
    elif any(str(Path(run['initial_model']).resolve()).startswith(prefix) for prefix in ('/dataset/',)):
        raise ValueError('Official root-disk dependency staging still pending')
    for name,sha in run['input_sha256'].items():add(base/'data'/name,sha,'canonical_input')
    source_manifest=base/'preparation/source-files.json'
    add(source_manifest,digest(source_manifest),'source_file_manifest')
    for name,sha in json.loads(source_manifest.read_text()).items():
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Source file escapes repository')
        add(base/'repo'/relative,sha,'frozen_source')
    return dict(schema_version=1,status='BACKUP_INVENTORY_ONLY',run_id=pair['run_id'],step=step,
        source_base=str(base),paired_receipt_sha256=expected_receipt_sha,source_commit=run['source_commit'],
        files=sorted(rows.values(),key=lambda row:row['source_path']),bytes=sum(row['bytes'] for row in rows.values()),source_aliases=aliases,
        independent_full_sha_performed=False,semantic_reload_performed=False,all_experiment_backup_complete=False,
        remaining=['independent filesystem copy and source/target complete SHA','actual semantic reload',
                   'retrieval assets and terminal raw logs/rollouts/final evaluation archival'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--receipt',required=True);parser.add_argument('--receipt-sha256',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();plan=inventory(args.receipt,args.receipt_sha256)
    with Path(args.output).open('x') as f:json.dump(plan,f,indent=2)
    print(json.dumps({k:v for k,v in plan.items() if k!='files'}))


if __name__=='__main__':main()
