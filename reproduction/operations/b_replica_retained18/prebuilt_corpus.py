"""Read already SHA-qualified Arrow shards in their original numeric order."""
import json,hashlib,os,subprocess
from pathlib import Path

def identity(path):
    s=Path(path).stat();return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns]

def load(cache):
    import datasets
    cache=Path(cache);proof=json.loads((cache/'prebuilt-arrow.json').read_text())
    if proof['schema']!='retained_full_sha_original_arrow_v1':raise ValueError('Unqualified Arrow binding')
    rows=proof['files'];names=[Path(r['path']).name for r in rows]
    if names!=['json-train-%05d-of-%05d.arrow'%(i,len(rows))for i in range(len(rows))]:
        raise ValueError('Preserve complete original Arrow shard ordering')
    translations=[]
    for row in rows:
        actual=identity(row['path'])
        if actual==row['identity']:continue
        # st_dev is a client-local filesystem number. The exact same retained18
        # inode exported over NFS has a different st_dev on another H100 node.
        # Never relax inode/size/data-mtime, shard order, or the prior full SHA.
        if actual[1:]!=row['identity'][1:]:raise ValueError('Qualified Arrow identity changed')
        source,fstype=subprocess.check_output(['findmnt','-T',row['path'],'-n','-o','SOURCE,FSTYPE'],text=True).split()
        if fstype!='nfs4' or source not in ('10.33.0.18:/mnt/data','10.33.0.18:/mnt/data/shaoshitong'):
            raise ValueError('Device translation requires the exact retained18 NFS export')
        if len(row.get('sha256',''))!=64:raise ValueError('Missing original full-file SHA qualification')
        translations.append(dict(path=row['path'],source_identity=row['identity'],client_identity=actual,original_full_sha256=row['sha256'],nfs_source=source))
    if translations:
        audit=Path(os.environ['SEARCH_G1_BASE'])/'retrieval-cache/prebuilt-nfs-device-identity.json';audit.parent.mkdir(parents=True,exist_ok=True)
        value=dict(original_proof_sha256=hashlib.sha256((cache/'prebuilt-arrow.json').read_bytes()).hexdigest(),translations=translations,new_byte_readback_claimed=False)
        if audit.exists():
            if json.loads(audit.read_text())!=value:raise ValueError('Pinned NFS identity translation changed')
        else:
            with audit.open('x')as f:json.dump(value,f,sort_keys=True)
    info=datasets.DatasetInfo.from_dict(proof['dataset_info'])
    result=datasets.concatenate_datasets([datasets.Dataset.from_file(r['path'],info=info,split='train')for r in rows])
    if len(result)!=proof['documents'] or len(result)!=21015324:raise ValueError('Original corpus count differs')
    return result
