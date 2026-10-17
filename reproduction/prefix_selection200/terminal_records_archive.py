"""Archive sealed original records after native200/final QA/owned cleanup.

No GPU, model loading, source deletion, training changes or healthy-run gates.
Model checkpoints and fixed assets have their own independent copies. This
closes the remaining original trajectories, evaluation outputs, logs and code,
including authentic source paths inherited across successful recoveries.
"""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time

try:
    from .checkpoint_backup_native import (MOUNT, PARENT, RUN, RATE, RESERVE,
        exported, full_read, need, read, record, sha, utc)
except ImportError:
    from checkpoint_backup_native import (MOUNT, PARENT, RUN, RATE, RESERVE,
        exported, full_read, need, read, record, sha, utc)

NAME = RUN[:12]+'-terminal-records-once-v1'
PASS = 'INDEPENDENT_TERMINAL_RECORDS_FULL_SHA_PASS'


def assert_terminal(config, joint, pipeline, cleanup, proof):
    """A supplied terminal receipt must bind the actual sealed run, not intent."""
    need(config['run_id'] == joint['run_id'] == pipeline['run_id'] == cleanup['run_id'] == proof['run_id'] == RUN,
         'Terminal identities differ')
    need(config['outers'] == joint['outer'] == pipeline['complete_outers'] == 200,
         'All200 actually complete before archival')
    need(joint['actor']['actual_adam_steps_per_rank'] == [4000]*8
         and joint['probe']['actual_adam_steps_per_rank'] == [1000]*8
         and joint['actor']['scheduler_completed_outer'] == 200, 'Final optimizer/scheduler clocks differ')
    need(proof['status'] == 'NATIVE_COMPLETED_ZERO_OWNED_RELEASE_PROVED'
         and proof['source_base'] == config['base']
         and proof['native']['JobState'] == 'COMPLETED' and proof['native']['ExitCode'] == '0:0',
         'Actual native COMPLETED0:0 proof required')
    release = proof['owned_release']
    need(release['two_actual_checks'] and release['all_original_births_gone']
         and release['cgroup_gone'] and release['own_gpu_processes'] == [], 'Independent release evidence incomplete')
    # The original cleanup receipt can observe a just-exited zombie. Preserve
    # it; the later two actual release observations are authoritative.
    need({row['role'] for row in proof['evidence']} >=
         {'native_terminal','owned_release_first','owned_release_second'}, 'Original terminal/release evidence is missing')


def canonical_rows(original):
    rows = []
    for ordinal, row in enumerate(original):
        qid = row.get('qid', row.get('__g1_question_id'))
        if qid is None:
            qid = json.dumps([str(row['data_source']), str((row.get('extra_info') or {}).get('index', ordinal))])
        rows.append(dict(qid=str(qid), evaluation_ordinal=ordinal, data_source=row['data_source']))
    need(len({r['qid'] for r in rows}) == len(rows), 'Canonical question identities repeat')
    return rows


def assert_final_rows(canonical, worker_rows, result, *, expected_count=51713, expected_sources=7):
    """Compare every actual ordinal/qid/source and all seven source counts."""
    need(len(canonical) == result['qa_count'] == expected_count, 'Final QA count differs')
    expected = Counter(r['data_source'] for r in canonical)
    need(len(expected) == expected_sources and
         {k:v['qa_count'] for k,v in result['per_source'].items()} == dict(expected),
         'Final source identities/counts differ from the canonical input')
    actual = sorted(worker_rows, key=lambda r:r['evaluation_ordinal'])
    need(len(actual) == expected_count and len({r['qid'] for r in actual}) == expected_count,
         'Final worker rows omitted or duplicated a question')
    for wanted, got in zip(canonical, actual):
        need(all(got[key] == wanted[key] for key in ('qid','data_source','evaluation_ordinal')),
             'Final output changed a canonical ordinal, question or source')
        need(got['em'] in (0,1) and math.isfinite(got['f1']) and 0<=got['f1']<=1,
             'An actual final answer score is invalid')
    for key in ('em','f1'):
        need(abs(sum(r[key]for r in actual)/expected_count-result[key])<=1e-12,'Final aggregate score differs from its original worker rows')
        for name,count in expected.items():
            need(abs(sum(r[key]for r in actual if r['data_source']==name)/count-result['per_source'][name][key])<=1e-12,
                 'Final per-source score differs from its original worker rows')
    return dict(expected)


def allowed_base(path):
    path = Path(path)
    need(path.parent == Path('/mnt/data/shaoshitong') and '..' not in path.parts
         and path.name.startswith('search-g1-prefix-selection200-actorhidden20-')
         and path.resolve() == path, 'Only this experiment source namespace is admitted')
    return path


class FileInventory:
    """Deduplicate actual hard links while preserving every original path."""
    def __init__(self, bases):
        self.bases = [Path(p) for p in bases]
        self.rows = {}

    def add(self, path, role, expected_sha256=None):
        path = Path(path)
        need(path.is_absolute() and '..' not in path.parts
             and any(base in path.parents for base in self.bases), 'Archive path escapes its authenticated run')
        need(path.is_file() and not path.is_symlink() and path.resolve() == path, 'A real immutable original file is required')
        need(path.name not in ('auth.json','.env','id_rsa','id_ed25519') and path.suffix != '.pem',
             'Credentials are outside archival scope')
        st = path.stat(); key = st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns
        if key not in self.rows:
            self.rows[key] = dict(source_path=str(path), original_paths=[], roles=[], bytes=st.st_size,
                source_mtime_ns=st.st_mtime_ns, expected_sha256=expected_sha256,
                source_mode=st.st_mode & 0o777,
                relative_path=str(Path('payload')/path.relative_to('/')))
        row = self.rows[key]
        if expected_sha256 is not None:
            need(row['expected_sha256'] in (None, expected_sha256), 'Conflicting digests for the same original inode')
            row['expected_sha256'] = expected_sha256
        if str(path) not in row['original_paths']: row['original_paths'].append(str(path))
        if role not in row['roles']: row['roles'].append(role)

    def tree(self, root, role, *, actor_records=False, json_only=False):
        root = Path(root)
        if not root.exists(): return
        for path in sorted(root.rglob('*')):
            if any(part in ('__pycache__','.git') for part in path.relative_to(root).parts): continue
            if not path.is_file(): continue
            if actor_records and ('huggingface' in path.parts or path.suffix=='.safetensors' or re.fullmatch(r'rank-\d+\.pt',path.name)): continue
            if path.name == 'probe-state-at-completion.pt' or re.search(r'-actor-hidden\.npy(?:\.(?:partial|tmp))?$',path.name):
                continue  # Declared consumed derived buffers/model snapshots are covered separately.
            if role=='source_preparation' and (path.suffix in('.pt','.bin','.safetensors','.npy','.pkl','.whl') or path.name.endswith(('.tar','.tar.gz'))):
                continue  # Preparation copies/packages are not original trajectory records.
            if json_only and path.suffix not in ('.json','.jsonl','.log','.txt','.gz'): continue
            if path.name.startswith('terminal-records-inventory-'): continue
            # A failed historical attempt can leave a closed .partial file.
            # Preserve it as evidence; terminal release + stable full reads,
            # rather than its filename, establish that copying is safe.
            self.add(path, role)

    def files(self):
        return sorted(self.rows.values(),key=lambda r:r['relative_path'])


def inventory(base, proof_path, proof_sha256):
    need(socket.gethostname() == 'h100-8', 'Prepare sealed records on the source node')
    base = allowed_base(base); proof_path = Path(proof_path)
    need(base in proof_path.parents and sha(proof_path) == proof_sha256, 'Pinned actual terminal proof differs')
    c = read(base/'preparation/launch-config.json'); joint = read(base/'latest-joint.json')
    pipeline = read(base/'logs/pipeline-complete.json'); cleanup = read(base/'logs/cleanup-status.json')
    proof = read(proof_path); assert_terminal(c,joint,pipeline,cleanup,proof)
    submitted=read(base/'preparation/submit-receipt.json');intent=read(base/'preparation/submit-intent.json')
    need(submitted['status']=='SUBMITTED' and submitted['returncode']==0
         and submitted['intent_sha256']==sha(base/'preparation/submit-intent.json')
         and intent['run_id']==RUN and str(proof['native']['JobId'])==str(submitted['job_id'])
         and proof['native']['JobName']==intent['job_name'], 'Native exit proof is not the actual submitted job')
    need(read(base/'logs/qa-pending.json') == {}, 'Actual scheduled QA failures remain pending')
    sys.path.insert(0,str(base/'repo'))
    from reproduction.prefix_selection200.selection_metrics import summarize_outer

    bases = {base}; points = []; metadata = []; saved_by_outer = {}
    for outer in range(1,201):
        path = base/'outers'/('outer-%06d'%outer)/'complete.json'; saved = read(path)
        need(saved['run_id'] == RUN and saved['outer'] == outer
             and saved['actor']['actual_adam_steps_per_rank'] == [20*outer]*8
             and saved['probe']['actual_adam_steps_per_rank'] == [100*(outer//20)]*8,
             'A logical outer lacks its correct completed optimizer boundary')
        source = allowed_base(Path(saved['rollout']['own_em']['path']).parent.parent.parent.parent)
        source_outer = Path(saved['rollout']['own_em']['path']).parent.parent
        need(source_outer.parent.parent == source and source_outer.name == 'outer-%06d'%outer,
             'Inherited original record is not the same logical outer')
        actual = read(source_outer/'complete.json')
        need(actual['run_id'] == RUN and actual['outer'] == outer
             and actual['selection_sha256'] == saved['selection_sha256'], 'Inherited joint identity differs')
        row = summarize_outer(source_outer/'complete.json')
        need(row['actor_actual_adam_per_rank'] == [20*outer]*8, 'Actual inherited actor boundary differs')
        bases.add(source); points.append((path,source_outer)); metadata.append(row)
        saved_by_outer[outer] = saved
    # Follow only the authenticated recovery chain, preserving failed-run logs
    # and frozen sources without copying obsolete actor weight trees again.
    pending = list(bases)
    while pending:
        source = pending.pop(); cfg = read(source/'preparation/launch-config.json')
        need(cfg['run_id'] == RUN, 'History belongs to another experiment')
        parent = cfg.get('resume_joint',{}).get('source_base')
        if parent:
            parent = allowed_base(parent)
            if parent not in bases: bases.add(parent); pending.append(parent)
    files = FileInventory(bases)
    for path,source_outer in points:
        files.add(path,'logical_joint_receipt',sha(path)); files.tree(source_outer,'original_question_prefix_trajectory_and_metrics')
    for source in sorted(bases):
        for name in ('logs','preparation','validation-retries'):
            files.tree(source/name,'source_'+name)
        files.tree(source/'outers','original_records_including_superseded_attempts')
        files.tree(source/'actor','actor_ledgers_and_driver_records',actor_records=True)
        files.tree(source/'grounding','grounding_request_records',json_only=True)
        manifest = source/'preparation/source-files.json'
        for name,digest in read(manifest).items():
            relative = Path(name);need(not relative.is_absolute() and '..' not in relative.parts,'Frozen source path escapes repository')
            files.add(source/'repo'/relative,'frozen_source',digest)
    files.tree(base/'question-pool','final_pool_state_and_permutation')
    files.tree(base/'probe','final_probe_metadata',json_only=True)
    files.tree(base/'final-evaluation','full_final_evaluation')

    final = read(base/'final-evaluation/complete.json')
    need(final['actor_version'] == joint['actor']['actor_version'] and final['evaluation_contract']['kind'] == 'legacy_pre_d00_async_fp32'
         and final['actual_worker_exit_codes'] == {str(i):0 for i in range(8)}, 'Actual final actor/protocol/worker exits differ')
    need(sha(c['final_questions']) == c['final_questions_sha256'] == final['question_source_sha256'], 'Canonical final input changed')
    import pyarrow.parquet as pq
    canonical = canonical_rows(pq.read_table(c['final_questions']).to_pylist())
    rows = []

    def evaluation_artifacts(result):
        worker_rows=[]
        need(len(result['output_shards'])==len(result['rank_receipts'])==8,'Eight actual QA shards/receipts required')
        for artifact in result['output_shards']:
            files.add(artifact['path'],'full_actual_evaluation_tokens',artifact['sha256'])
        for receipt in result['rank_receipts']:
            files.add(receipt['path'],'actual_evaluation_worker_receipt',receipt['sha256'])
            worker=read(receipt['path'])
            need(worker['actor_version']==result['actor_version'] and worker['evaluation_contract']==result['evaluation_contract']
                 and worker['actual_compute_dtype']=='torch.float32','Actual QA worker actor/FP32 contract differs')
            worker_rows.extend(worker['rows'])
        return worker_rows
    rows=evaluation_artifacts(final)
    sources = assert_final_rows(canonical,rows,final)
    need(sha(c['validation_questions'])==c['validation_questions_sha256'],'Original fixed512 input changed')
    validation_rows=canonical_rows(pq.read_table(c['validation_questions']).to_pylist())
    need(Counter(r['data_source']for r in validation_rows)==dict(nq=256,hotpotqa=256),'Fixed512 source split changed')
    # Scheduled validation may have succeeded in a retry namespace. Resolve
    # its original full-shard receipts; never substitute final metrics for it.
    qa = {}
    for outer in (5,50,100,150,200):
        candidates = []
        for source in sorted(bases):
            point = source/'outers'/('outer-%06d'%outer)/'validation/complete.json'
            if point.is_file(): candidates.append(point)
            candidates += list((source/'validation-retries').glob('outer-%06d-*/complete.json'%outer))
        expected_model = saved_by_outer[outer]['actor']['hf_model_dir']
        matching = [(p,read(p)) for p in candidates
                    if read(p).get('actor_version') == RUN+'-actor%04d'%outer
                    and read(p).get('actor_model_dir') == expected_model]
        need(matching, 'Scheduled validation original receipt missing')
        path,value = matching[-1]
        need(value['qa_count'] == 512 and value['actual_worker_exit_codes'] == {str(i):0 for i in range(8)}
             and value['evaluation_contract'] == final['evaluation_contract']
             and value['question_source_sha256'] == c['validation_questions_sha256'], 'Scheduled validation did not actually succeed')
        files.add(path,'successful_scheduled_QA',sha(path))
        assert_final_rows(validation_rows,evaluation_artifacts(value),value,expected_count=512,expected_sources=2)
        qa[str(outer)] = dict(path=str(path),sha256=sha(path))
    files.add(proof_path,'actual_native_exit_and_independent_owned_release',proof_sha256)
    for item in proof['evidence']:
        need(sha(item['path']) == item['sha256'],'Terminal original evidence changed')
        files.add(item['path'],'native_terminal_original_evidence',item['sha256'])
    need(sha(proof_path) == proof_sha256,'Terminal proof changed during inventory')
    payloads = files.files()
    return dict(status='SEALED_TERMINAL_RECORDS_INVENTORY_ONLY',run_id=RUN,source_base=str(base),
        terminal_proof_sha256=proof_sha256,created_utc=utc(),files=payloads,
        bytes=sum(r['bytes'] for r in payloads),original_path_count=sum(len(r['original_paths']) for r in payloads),
        original_hard_links_deduplicated=True,complete_outers=200,final_qa_count=51713,
        final_exact_source_counts=sources,scheduled_qa=qa,logical_outer_metrics=metadata,
        independent_backup_performed=False,training_modified=False,semantic_reload_performed=False)


def dependencies_pass():
    for outer in (5,50,100,150,200):
        root=PARENT/(RUN[:12]+'-outer-%04d-once-v1'%outer)
        result=read(root/'RESULT.json')
        need(result['status']=='INDEPENDENT_PREFIX200_JOINT_BACKUP_FULL_SHA_PASS' and result['run_id']==RUN
             and result['outer']==outer,'Normal independent checkpoint copy is unfinished')
    fixed=read(PARENT/(RUN[:12]+'-fixed-recovery-assets-once-v1')/'RESULT.json')
    need(fixed['status']=='INDEPENDENT_FIXED_RECOVERY_ASSETS_FULL_SHA_PASS' and fixed['run_id']==RUN,
         'Fixed recovery dependency copy is unfinished')


def worker(root):
    root=Path(root);need(socket.gethostname()=='h100-10' and root==PARENT/NAME and root.resolve()==root,'Wrong archival destination')
    intent=read(root/'INTENT.json');need(sha(__file__)==intent['worker_sha256'] and sha(Path(__file__).with_name('checkpoint_backup_native.py'))==intent['common_sha256'],'Published archive control changed')
    dependencies_pass();path=exported(intent['inventory_path']);need(sha(path)==intent['inventory_sha256'],'Sealed inventory changed')
    plan=read(path);need(plan['run_id']==RUN and plan['status']=='SEALED_TERMINAL_RECORDS_INVENTORY_ONLY' and plan['complete_outers']==200,'Wrong sealed run')
    need(subprocess.check_output(['findmnt','-n','-T',str(root),'-o','FSTYPE'],text=True).strip()=='ext4','Independent local ext4 required')
    need(os.statvfs(root).f_bavail*os.statvfs(root).f_frsize>=plan['bytes']+RESERVE,'Ordinary free reserve required')
    os.nice(19);staging=root/'records.partial';staging.mkdir();copied=[]
    for row in plan['files']:
        source=exported(row['source_path']);relative=Path(row['relative_path'])
        need(not relative.is_absolute() and '..' not in relative.parts and relative.parts[0]=='payload','Invalid payload path')
        need(source.resolve().is_relative_to(MOUNT.resolve()) and source.stat().st_dev!=root.stat().st_dev,'Source is not the distinct data export')
        need(source.stat().st_size==row['bytes'] and source.stat().st_mtime_ns==row['source_mtime_ns'],'Sealed original metadata changed')
        need(os.statvfs(root).f_bavail*os.statvfs(root).f_frsize>=row['bytes']+RESERVE,'Ordinary free reserve exhausted')
        target=staging/relative;target.parent.mkdir(parents=True,exist_ok=True)
        def progress(count,phase='source_copy'):
            record(root,'PROGRESS.json',dict(utc=utc(),current_file=str(relative),current_file_bytes=count,current_phase=phase,
                completed_files=len(copied),expected_files=len(plan['files']),completed_bytes=sum(x['bytes']for x in copied),expected_bytes=plan['bytes']))
        with target.open('xb')as out:
            count,digest=full_read(source,out,progress);out.flush();os.fsync(out.fileno())
        need(count==row['bytes'] and row['expected_sha256'] in(None,digest),'Original full SHA differs')
        independent=full_read(target,callback=lambda n:progress(n,'independent_target_full_sha'))
        need(independent==(count,digest),'Independent archival target SHA differs')
        os.chmod(target,row['source_mode'])
        copied.append(dict(row,source_full_sha256=digest,destination_full_sha256=independent[1]));progress(count,'verified')
    need(sha(path)==intent['inventory_sha256'],'Inventory changed during archive')
    inventory_raw=path.read_bytes()
    need(hashlib.sha256(inventory_raw).hexdigest()==intent['inventory_sha256'],'Sealed inventory bytes changed')
    with(staging/'INVENTORY.json').open('xb')as out:out.write(inventory_raw);out.flush();os.fsync(out.fileno())
    need(sha(staging/'INVENTORY.json')==intent['inventory_sha256'],'Independent inventory metadata copy differs')
    mapping={name:r['relative_path']for r in copied for name in r['original_paths']}
    record(staging,'RESTORE_MAPPING.json',dict(run_id=RUN,mapping=mapping))
    for directory in sorted([staging]+[p for p in staging.rglob('*')if p.is_dir()],key=lambda p:len(p.parts),reverse=True):
        fd=os.open(directory,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
    need(not(root/'records').exists(),'Do not overwrite a previous archive');os.rename(staging,root/'records')
    result=dict(status=PASS,run_id=RUN,completed_utc=utc(),files=copied,bytes=plan['bytes'],
        original_path_count=len(mapping),inventory_sha256=intent['inventory_sha256'],
        restore_mapping_sha256=sha(root/'records/RESTORE_MAPPING.json'),source_bases_preserved=True,
        generated_metadata_files=['INVENTORY.json','RESTORE_MAPPING.json'],
        training_modified=False,model_bytes_transit_Mac=False,semantic_reload_performed=False,
        all_experiment_backup_complete=False,note='Final independent closure audit remains a separate requirement')
    record(root,'RESULT.json',result);return result


def launch(root, inventory_path, inventory_sha256, code_commit):
    """Called on10 only after actual native0 and all prerequisite copies pass."""
    root=Path(root);need(socket.gethostname()=='h100-10' and root==PARENT/NAME
        and root.parent.resolve()==root.parent,'Wrong archive namespace')
    need(re.fullmatch(r'[0-9a-f]{40}',code_commit) is not None,'Published full source commit required')
    with(PARENT/'terminal-records-start.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if root.exists():return dict(status='EXISTING_NAMESPACE_RECONCILE_NO_REPLAY',namespace=str(root))
        dependencies_pass();path=exported(inventory_path)
        need(sha(path)==inventory_sha256,'Prepared inventory SHA differs')
        plan=read(path);need(plan['run_id']==RUN and plan['status']=='SEALED_TERMINAL_RECORDS_INVENTORY_ONLY'
            and plan['complete_outers']==200 and plan['final_qa_count']==51713,'Actual sealed experiment required')
        need(os.statvfs(PARENT).f_bavail*os.statvfs(PARENT).f_frsize>=plan['bytes']+RESERVE,'Ordinary free reserve required')
        root.mkdir(mode=0o700)
        for name in('terminal_records_archive.py','checkpoint_backup_native.py'):
            code=Path(__file__).with_name(name).read_bytes()
            with(root/name).open('xb')as f:f.write(code);f.flush();os.fsync(f.fileno())
            need(sha(root/name)==hashlib.sha256(code).hexdigest(),'Published control write differs')
        intent=dict(run_id=RUN,inventory_path=str(inventory_path),inventory_sha256=inventory_sha256,
            worker_sha256=sha(root/'terminal_records_archive.py'),common_sha256=sha(root/'checkpoint_backup_native.py'),
            code_commit=code_commit,expected_files=len(plan['files']),expected_bytes=plan['bytes'],created_utc=utc())
        with(root/'INTENT.json').open('x')as f:json.dump(intent,f,sort_keys=True);f.flush();os.fsync(f.fileno())
        with(root/'archive.log').open('xb')as log:
            child=subprocess.Popen([sys.executable,'-E','-B','-S',str(root/'terminal_records_archive.py'),'worker','--root',str(root)],
                stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True,
                env={'PATH':'/usr/bin:/bin','LANG':'C','LC_ALL':'C','CUDA_VISIBLE_DEVICES':''})
        fields=(Path('/proc')/str(child.pid)/'stat').read_text().rsplit(')',1)[1].split()
        receipt=dict(status='TERMINAL_RECORD_COPY_STARTED_NOT_COMPLETE',run_id=RUN,namespace=str(root),
            pid=child.pid,start_ticks=int(fields[19]),sid=os.getsid(child.pid),expected_files=len(plan['files']),
            expected_bytes=plan['bytes'],bytes_per_second_limit=RATE,ordinary_reserve_bytes=RESERVE,
            semantic_reload_performed=False,training_modified=False,model_bytes_transit_Mac=False)
        with(root/'RECEIPT.json').open('x')as f:json.dump(receipt,f,sort_keys=True);f.flush();os.fsync(f.fileno())
        return receipt


def status(root, final_readback=False):
    """Only small summaries leave10; the full original-data inventory stays remote."""
    root=Path(root);need(socket.gethostname()=='h100-10' and root==PARENT/NAME,'Wrong archive status namespace')
    result=dict(namespace=str(root),exists=root.exists(),read_only=True)
    if not root.exists():return result
    for name in('INTENT.json','RECEIPT.json','PROGRESS.json','ERROR.json'):
        if(root/name).exists():result[name]=read(root/name)
    if 'RECEIPT.json'in result:
        receipt=result['RECEIPT.json'];proc=Path('/proc')/str(receipt['pid'])/'stat'
        fields=proc.read_text().rsplit(')',1)[1].split()if proc.exists()else None
        same=fields is not None and int(fields[19])==receipt['start_ticks']
        result['process']=dict(pid=receipt['pid'],same_birth=same,original_birth_gone=not same,state=fields[0]if same else None)
    if(root/'RESULT.json').exists():
        value=read(root/'RESULT.json')
        result['result_summary']={k:v for k,v in value.items()if k!='files'}
        result['result_summary'].update(payload_file_count=len(value['files']),result_full_read_sha256=sha(root/'RESULT.json'),
            all_recorded_source_target_full_sha_equal=all(r['source_full_sha256']==r['destination_full_sha256']for r in value['files']))
        if final_readback:
            final=root/'records';mapping={name:r['relative_path']for r in value['files']for name in r['original_paths']}
            actual=[(final/r['relative_path'])for r in value['files']]
            result['final_readback']=dict(all_payloads_regular_correct_size=all(p.is_file()and not p.is_symlink()and p.stat().st_size==r['bytes']for p,r in zip(actual,value['files'])),
                payload_bytes=sum(p.stat().st_size for p in actual),physical_payload_files=len(actual),
                actual_all_file_count=sum(1 for p in final.rglob('*')if p.is_file()),
                inventory_metadata_sha_matches=sha(final/'INVENTORY.json')==value['inventory_sha256'],
                restore_mapping_sha_matches=sha(final/'RESTORE_MAPPING.json')==value['restore_mapping_sha256'],
                restore_mapping_exact=read(final/'RESTORE_MAPPING.json')['mapping']==mapping,
                staging_absent=not(root/'records.partial').exists(),large_payloads_rehashed=False)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='phase',required=True)
    i=sub.add_parser('inventory');i.add_argument('--base',required=True);i.add_argument('--terminal-proof',required=True);i.add_argument('--terminal-proof-sha',required=True);i.add_argument('--output',required=True)
    w=sub.add_parser('worker');w.add_argument('--root',required=True)
    start=sub.add_parser('launch');start.add_argument('--root',required=True);start.add_argument('--inventory-path',required=True);start.add_argument('--inventory-sha',required=True);start.add_argument('--code-commit',required=True)
    check=sub.add_parser('status');check.add_argument('--root',required=True);check.add_argument('--final-readback',action='store_true');a=p.parse_args()
    if a.phase=='inventory':
        output=Path(a.output);base=allowed_base(a.base);need(base/'preparation' in output.parents and not output.exists(),'Fresh source preparation inventory required')
        value=inventory(base,a.terminal_proof,a.terminal_proof_sha);record(output.parent,output.name,value)
        print(json.dumps({k:v for k,v in value.items()if k not in('files','logical_outer_metrics')}|dict(output=str(output),sha256=sha(output))))
    elif a.phase=='launch':print(json.dumps(launch(a.root,a.inventory_path,a.inventory_sha,a.code_commit)))
    elif a.phase=='status':print(json.dumps(status(a.root,a.final_readback)))
    else:
        try:worker(a.root)
        except BaseException as error:
            record(Path(a.root),'ERROR.json',dict(status='INCOMPLETE_PRESERVE_SOURCE_AND_PARTIAL',error=repr(error),utc=utc()));raise
