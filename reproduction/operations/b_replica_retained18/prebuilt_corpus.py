"""Read already SHA-qualified Arrow shards in their original numeric order."""
import json
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
    for row in rows:
        if identity(row['path'])!=row['identity']:raise ValueError('Qualified Arrow bytes changed')
    info=datasets.DatasetInfo.from_dict(proof['dataset_info'])
    result=datasets.concatenate_datasets([datasets.Dataset.from_file(r['path'],info=info,split='train')for r in rows])
    if len(result)!=proof['documents'] or len(result)!=21015324:raise ValueError('Original corpus count differs')
    return result
