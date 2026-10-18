"""Zero-write failed-start repair; retain initialized raw/EMA and all evidence."""
import hashlib,json,os,pathlib,time,uuid
from reproduction.prefix20_online.launch import once,read,sha
from reproduction.a800.owned_processes import same

def full_sha(p):
    h=hashlib.sha256()
    with pathlib.Path(p).open('rb')as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()

def prepare(source,target,canonical,commit):
    from .prebuilt_corpus import identity
    source=pathlib.Path(source);target=pathlib.Path(target)
    old=read(source/'preparation/launch-config.json');cleanup=read(source/'logs/cleanup-status.json')
    assert cleanup['owned_births_all_gone'] and not same(read(source/'preparation/submit-receipt.json')['leader'])
    assert not(source/'actor').exists()and not(source/'question-pool').exists()and not(source/'latest-joint.json').exists()
    assert 'Retained corpus cache belongs to different input'in(source/'logs/retriever.log').read_text()
    head=read(source/'probe/latest.json');assert head['actual_adam_steps_per_rank']==[0]*4 and head['completed_outer']==0
    # A new output/source directory has already been SHA-staged by the controller.
    assert not(target/'preparation/launch-config.json').exists()
    def remap(v):
        if isinstance(v,str):return canonical+v[len(old['base']):]if v==old['base']or v.startswith(old['base']+'/')else v
        if isinstance(v,dict):return {k:remap(x)for k,x in v.items()}
        if isinstance(v,list):return [remap(x)for x in v]
        return v
    c=remap(old);c.update(source_commit=commit,execution_id=uuid.uuid4().hex,retrieval_data_cache=canonical+'/qualified-corpus-cache')
    fixed=pathlib.Path('/mnt/data/shaoshitong/search-g1-prefix5-h10011-retained-fixed-assets-20261018')
    assets=fixed/'shaoshitong';cache=assets/'search-g1-prefix20-online1000-20261016-ca1c60f/retrieval-cache'
    archive=assets/'search-r1-rl/gigpo/assets-root/assets/wiki/wiki-18.jsonl'
    rows=read(fixed/'cache2-completed-files.json')
    def expected(f):
        suffix='/shaoshitong/'+str(f.relative_to(assets))
        matches=[r for r in rows if r['target'].endswith(suffix)];assert len(matches)==1
        r=matches[0];return r.get('proven_sha256',r.get('destination_sha256'))
    archive_sha=full_sha(archive);assert archive_sha==expected(archive)
    extraction=read(cache/'corpus.json');content=cache/'wiki_dump.jsonl'
    assert full_sha(content)==extraction['content_sha256']==expected(content)
    private=target/'qualified-corpus-cache';private.mkdir();os.link(content,private/'wiki_dump.jsonl')
    extraction.update(source=c['retrieval_corpus'],source_size=archive.stat().st_size,source_mtime_ns=archive.stat().st_mtime_ns)
    once(private/'corpus.json',extraction)
    infos=list((cache/'arrow').rglob('dataset_info.json'));assert len(infos)==1
    info=read(infos[0]);assert info['splits']['train']['num_examples']==21015324
    shards=sorted(infos[0].parent.glob('json-train-*.arrow'));assert len(shards)==28
    (private/'arrow').mkdir();qualified=[]
    for shard in shards:
        digest=full_sha(shard);assert digest==expected(shard),str(shard)
        dst=private/'arrow'/shard.name;os.link(shard,dst)
        qualified.append(dict(path=canonical+'/qualified-corpus-cache/arrow/'+shard.name,identity=identity(dst),sha256=digest))
    once(private/'prebuilt-arrow.json',dict(schema='retained_full_sha_original_arrow_v1',documents=21015324,dataset_info=info,files=qualified,archive_sha256=archive_sha,corpus_sha256=extraction['content_sha256'],checked_unix=time.time()))
    (target/'probe').mkdir()
    for name in ['latest.pt','latest.json','head-outer-0000.pt','initialization-complete.json','initialization-intent.json']:os.link(source/'probe'/name,target/'probe'/name)
    assert sha(target/'probe/latest.pt')==head['sha256']
    once(target/'preparation/launch-config.json',c)
    once(target/'preparation/zero-write-cache-repair.json',dict(source_base=str(source),source_cleanup=cleanup,run_id=c['run_id'],source_head=head,archive_sha256=archive_sha,content_sha256=extraction['content_sha256'],arrow_files=len(shards),documents=21015324,original_logs=[dict(path=str(f),sha256=sha(f))for f in(source/'logs').glob('*.log')],created_unix=time.time(),actor_actual_writes=0,probe_actual_writes=0,original_and_foreign_tasks_modified=False))
    return dict(base=canonical,run_id=c['run_id'],source_commit=commit,probe_preserved_sha256=head['sha256'],documents=21015324)
