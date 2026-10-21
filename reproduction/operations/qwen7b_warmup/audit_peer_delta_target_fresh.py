"""One independent fresh HPC3 archive/member readback of the sealed122 delta.

Gates on genuine receiver completion. Does not download, extract, alter source
assets, invoke GPU, recompute EM, read model features, or touch a live trainer.
"""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path,PurePosixPath
import resource
import time

B=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019')
SNAPSHOT='audited122-delta-20261021-673b8cb6'
TARGET=B/'peer-newseed-h100-20261021'/SNAPSHOT
OLD=B/'peer-newseed-h100-20261021/newseed-20261021-185747-445fc691'
OUTPUT=B/'algorithm-improvement-20261021/peer-audited122-target-independent-v1'
SOURCE_SHA='bb20a9e0f9e9dfea5ebdaf2ce6000d60d0c9ac0cdad0636b0afa9cd16f2733c4'
PLAN_SHA='3465c82e0d930569013b8d054e714f20cdb9f35972cbe7570829d13878e729da'
RECORD_SHA='673b8cb6abe7108e1d03a98b518121dafcd1ec10b156ec78f37c5faacfb7cc21'
RUN='70188b8e50fd4b45bc420ce58014f46e'
REVISION='a09a35458c702b33eeacc393d103063234e8bc28'
STATUS='PEER_NEWSEED_RAW_ALL_ARCHIVES_AND_FILES_FRESH_SHA256_VERIFIED'


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def read(path):return json.loads(Path(path).read_bytes())
def regular(path):
    path=Path(path)
    if path.resolve()!=path or not path.is_file():raise ValueError('Actual regular nonsymlink file required: '+str(path))
    return path

def proof(path):
    path=regular(path);before=path.stat();h=hashlib.sha256();count=0
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block);count+=len(block)
    after=path.stat()
    if (before.st_size,before.st_mtime_ns,before.st_ino)!=(after.st_size,after.st_mtime_ns,after.st_ino) or count!=before.st_size:raise ValueError('File changed during fresh SHA readback')
    return dict(path=str(path),bytes=count,sha256=h.hexdigest())

def checked(path,ref):
    actual=proof(path)
    if actual['bytes']!=ref['bytes'] or actual['sha256']!=ref['sha256']:raise ValueError('Fresh target SHA/bytes mismatch: '+str(path))
    return actual

def atomic(path,value):
    path=Path(path);temp=path.with_name(path.name+'.tmp')
    with temp.open('xb') as f:f.write(canonical(value)+b'\n');f.flush();os.fsync(f.fileno())
    temp.replace(path)

def gate(base):
    done=read(base/'complete.json');catalog=read(base/'catalog.json');binding=read(base/'catalog-binding.json')
    raw=binding['catalog_bytes'].encode()
    if (hashlib.sha256(raw).hexdigest()!=binding['catalog_sha256'] or json.loads(raw)!=catalog
            or done.get('status')!=STATUS or done['snapshot_id']!=base.name or done['target']!=str(base/'raw')
            or done['catalog_commit']!=binding['catalog_commit'] or done['catalog_sha256']!=binding['catalog_sha256']
            or done['all_archive_SHA256'] is not True or done['fresh_all_file_SHA256'] is not True
            or done['actor_updates']!=0 or done['probe_updates']!=0 or done['feature_bank_created'] is not False):
        raise ValueError('Actual source-bound complete HPC3 raw receipt required')
    source=catalog['source_snapshot'];groups=catalog['groups']
    if source['snapshot_id']!=base.name or source['source_node']!='h100-10' or source['revision']!=REVISION:raise ValueError('Official actual source snapshot identity differs')
    if source['source_run_id']!=RUN or source['seed']!=20261021 or source['actor_updates']!=0 or source['hidden_exported'] is not False:raise ValueError('Actual unchanged raw-only source required')
    if source['fresh_all_source_files_SHA256'] is not True or source['all_archives_fresh_SHA_and_member_SHA_verified'] is not True:raise ValueError('Source export fresh acceptance missing')
    original={g['group']:g for g in source['groups']}
    if len(original)!=len(groups):raise ValueError('Source/catalog distinct archive count differs')
    for g in groups:
        for key in ('group','archive','archive_bytes','archive_sha256','files','source_bytes','questions','trajectories'):
            if original[g['group']][key]!=g[key]:raise ValueError('Source manifest/catalog payload binding differs')
        accepted=read(base/'accepted'/(g['group']+'.json'))
        if (accepted['status']!='TARGET_ARCHIVE_AND_ALL_MEMBER_FILES_FRESH_SHA_VERIFIED' or accepted['catalog_sha256']!=binding['catalog_sha256']
                or accepted['archive_sha256']!=g['archive_sha256'] or accepted['archive_bytes']!=g['archive_bytes'] or accepted['files']!=len(g['files'])):raise ValueError('Receiver per-group completion binding differs')
    return done,catalog,binding

def identities(base,catalog):
    groups=[];refs=[]
    for record in catalog['groups']:
        cohort=[f for f in record['files'] if f['path'].endswith('/cohort.json')]
        if len(cohort)!=1:raise ValueError('Exactly one own cohort per sealed archive required')
        item=cohort[0];path=base/'raw'/item['path'];refs.append(checked(path,item));value=read(path)
        if (value['source_run_id']!=RUN or value['seed']!=20261021 or value['revision']!=REVISION
                or value['prefix_tokens']!=20 or value['hidden_exported'] is not False or value['questions']!=record['questions']):raise ValueError('Own immutable source G5 contract differs')
        for g in value['groups']:
            tids=g['trajectory_ids']
            if len(tids)!=5 or len(set(tids))!=5 or tids!=[RUN+'-outer000001-c%04d-g%d'%(g['candidate_ordinal'],j) for j in range(5)]:raise ValueError('Five own authentic trajectory IDs required')
            groups.append({k:g[k] for k in ('qid','question_hash','trajectory_ids')})
    qids={g['qid'] for g in groups};hashes={g['question_hash'] for g in groups};tids={t for g in groups for t in g['trajectory_ids']}
    if len(qids)!=len(groups) or len(tids)!=5*len(groups):raise ValueError('Source own IDs repeat')
    return groups,dict(qids=qids,hashes=hashes,tids=tids),refs

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUTPUT);a=p.parse_args()
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2]);resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3));resource.setrlimit(resource.RLIMIT_CPU,(120,120))
    if not (TARGET/'complete.json').is_file():print(json.dumps(dict(status='WAITING_FOR_ACTUAL_HPC3_RAW_COMPLETE',target=str(TARGET))));return
    a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'audit.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        destination=a.output/'complete.json'
        if destination.exists():print(json.dumps(dict(status='EXISTING_INDEPENDENT_FRESH_ACCEPTANCE_PRESERVED',receipt=proof(destination),result=read(destination))));return
        started=time.time();done,catalog,binding=gate(TARGET);source=catalog['source_snapshot']
        source_raw=canonical(source)+b'\n'
        if len(source_raw)!=403981 or hashlib.sha256(source_raw).hexdigest()!=SOURCE_SHA:raise ValueError('Exact frozen source complete differs from independently observed exporter receipt')
        if (done['groups']!=8 or done['files']!=1734 or done['archive_bytes']!=722391040 or done['questions']!=15616 or done['trajectories']!=78080
                or source['frozen_sealed_batches']!=122 or source['files']!=1734):raise ValueError('Frozen audited122 target budget differs')
        archives=[];members=[];seen=set();sourcebytes=0
        for record in catalog['groups']:
            name=record['group']
            if not name.startswith('group-') or Path(record['archive']).name!=record['archive'].split('/')[-1]:raise ValueError('Unsafe sealed archive name')
            archive=checked(TARGET/'archives'/Path(record['archive']).name,dict(bytes=record['archive_bytes'],sha256=record['archive_sha256']))
            archives.append(dict(group=name,**archive));own=[]
            for item in record['files']:
                rel=PurePosixPath(item['path'])
                if rel.is_absolute() or '..' in rel.parts or rel.parts[:2]!=(SNAPSHOT,name) or item['path'] in seen:raise ValueError('Unique safe exact source raw-member names required')
                seen.add(item['path']);own.append(checked(TARGET/'raw'/item['path'],item))
            if sum(v['bytes'] for v in own)!=record['source_bytes']:raise ValueError('Actual member source bytes differ')
            sourcebytes+=sum(v['bytes'] for v in own);members.extend(own)
            atomic(a.output/'progress.json',dict(stage='INDEPENDENT_ARCHIVE_AND_EVERY_RAW_MEMBER_FRESH_SHA_ADVANCING',groups=len(archives),files=len(members),archive_bytes=sum(v['bytes'] for v in archives),member_bytes=sourcebytes,epoch=time.time()))
        actual=set()
        for path in (TARGET/'raw').rglob('*'):
            if path.is_symlink():raise ValueError('Unexpected symlink in raw receive')
            if path.is_file():actual.add(str(path.relative_to(TARGET/'raw')))
        if actual!=seen or len(members)!=1734:raise ValueError('Declared/actual unpacked member inventory differs')
        plans=[f for g in catalog['groups'] for f in g['files'] if f['path'].endswith('/provenance/plan.json')]
        if len(plans)!=1:raise ValueError('Frozen subset plan missing')
        planpath=TARGET/'raw'/plans[0]['path'];planref=checked(planpath,plans[0]);plan=read(planpath)
        if planref['sha256']!=PLAN_SHA or plan['subset_batch_records_sha256']!=RECORD_SHA or len(plan['selected_batches'])!=122:raise ValueError('Exact precommitted122 source plan differs')
        # Only already accepted old snapshot's small cohort/plan metadata is read;
        # no old archive/raw rollout payload is scanned again.
        olddone,oldcatalog,oldbinding=gate(OLD)
        if olddone['questions']!=31232 or olddone['groups']!=16:raise ValueError('Original imported244 scope differs')
        newgroups,new,newrefs=identities(TARGET,catalog);oldgroups,old,oldrefs=identities(OLD,oldcatalog)
        overlap={key:len(new[key]&old[key]) for key in ('qids','hashes','tids')}
        oldplans=[f for g in oldcatalog['groups'] for f in g['files'] if f['path'].endswith('/provenance/plan.json')]
        if len(oldplans)!=1:raise ValueError('Old244 identity plan missing')
        oldplanref=checked(OLD/'raw'/oldplans[0]['path'],oldplans[0]);oldplan=read(oldplanref['path'])
        newinputs={r['input'] for r in plan['selected_batches']};oldinputs={r['input'] for r in oldplan['selected_batches']}
        overlap['batches']=len(newinputs&oldinputs)
        if len(newinputs)!=122 or len(oldinputs)!=244 or any(overlap.values()) or len(newgroups)!=15616 or len(oldgroups)!=31232:raise ValueError('New122/old244 overlap or actual counts differ')
        domain=dict(Counter(json.loads(g['qid'])[0] for g in newgroups))
        if domain!={'nq':12300,'hotpotqa':3316}:raise ValueError('Frozen actual question-domain identity differs')
        result=dict(status='AUDITED122_TARGET_ALL_ARCHIVES_AND_MEMBERS_FRESH_SHA_AND_OLD244_ZERO_OVERLAP_INDEPENDENTLY_PASSED',
            target=str(TARGET),receiver_complete=proof(TARGET/'complete.json'),catalog=proof(TARGET/'catalog.json'),binding=proof(TARGET/'catalog-binding.json'),
            source_complete=dict(bytes=len(source_raw),sha256=SOURCE_SHA,canonical_reconstruction_matches_exact_export_file=True),
            source_plan=planref,source_batch_records_sha256=RECORD_SHA,source_batches=122,questions=15616,trajectories=78080,
            archive_count=len(archives),archive_bytes=sum(v['bytes'] for v in archives),archives=archives,
            unpacked_files=len(members),unpacked_bytes=sourcebytes,all_archive_and_member_SHA_fresh=True,
            fresh_member_proofs_sha256=hashlib.sha256(canonical(members)).hexdigest(),actual_unpacked_inventory_exact=True,
            old244=dict(receiver_complete=proof(OLD/'complete.json'),catalog=proof(OLD/'catalog.json'),plan=oldplanref,cohort_proofs_sha256=hashlib.sha256(canonical(oldrefs)).hexdigest(),questions=len(oldgroups),batches=244),
            old244_overlap=overlap,domain_counts=domain,own_authentic_G5_IDs_verified=True,
            strict_EM_recomputed=False,strict_EM_ingest_owned_by_separate_operator=True,
            model_or_hidden_tensor_read=False,GPU_used=False,actor_updates=0,source_or_receiver_original_files_changed=False,
            old_bulk_archives_or_rollouts_rehashed=False,elapsed_seconds=time.time()-started,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,epoch=time.time())
        atomic(destination,result);print(json.dumps(dict(receipt=proof(destination),result=result)),flush=True)

if __name__=='__main__':main()
