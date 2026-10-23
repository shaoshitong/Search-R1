"""Admit the real localhost retrieval path against the actual GPU index proof."""
import hashlib,json,math,os,re,time
from pathlib import Path
from reproduction.common8.validation_identity import expected
from reproduction.common8.verify_data import digest
from reproduction.common8.stage_retrieval import atomic
from reproduction.common8.topology import execution_job_id


def compare_response(response, proof):
    rows=response['result']
    assert len(rows)==64 and len(proof['gpu_document_sha256'])==64
    hashes=[]
    for index,row in enumerate(rows):
        assert len(row)==3
        actual=[]
        for column,item in enumerate(row):
            assert set(item)=={'document','score'} and isinstance(item['document'],dict)
            score=item['score'];assert type(score) in (int,float) and math.isfinite(score)
            assert abs(score-proof['gpu_document_scores'][index][column])<=1e-4
            actual.append(hashlib.sha256(json.dumps(item['document'],sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest())
        assert actual==proof['gpu_document_sha256'][index], 'HTTP documents differ from actual GPU index result'
        hashes.append(actual)
    return hashes


def admit_http(base):
    import requests
    run=json.loads((base/'logs/run-manifest.json').read_text())
    proof=json.loads((base/'logs/retrieval-admission.json').read_text())
    assert proof['accepted'] is True and proof['run_id']==run['run_id']
    assert proof['slurm_job_id']==execution_job_id(run)
    record,frame,ids=expected(base,'stage')
    assert proof['stage_sha256']==record['sha256']==run['input_sha256']['stage.parquet']
    assert proof['question_ids']==ids
    queries=[]
    for chat in frame['prompt']:
        content=chat[0]['content'];marks=list(re.finditer(r'Question:\s*',content));assert marks
        queries.append(content[marks[-1].end():].strip())
    session=requests.Session();session.trust_env=False
    from search_g1.service_ports import retriever_url
    response=session.post(retriever_url(run),json={'queries':queries,'topk':3,'return_scores':True},timeout=300)
    response.raise_for_status();hashes=compare_response(response.json(),proof)
    result=dict(accepted=True,run_id=run['run_id'],slurm_job_id=execution_job_id(run),question_ids=ids,stage_sha256=record['sha256'],retrieval_admission_sha256=digest(base/'logs/retrieval-admission.json'),document_sha256=hashes,checked_unix=time.time(),scope='actual localhost HTTP64 document/score equality; not training or model accuracy')
    if run.get('execution_backend')=='ssh_local':
        result.update(execution_backend='ssh_local',execution_id=run['execution_id'])
    target=base/'logs/retrieval-http-admission.json';assert not target.exists();atomic(target,result)
    return result
