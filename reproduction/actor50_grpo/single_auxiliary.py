"""Explicit human-approved GPU4 retrieval mapping for the stopped full115 run."""
import hashlib
import json
from pathlib import Path

GPU4='GPU-c63282c6-1d05-224e-234b-2a8c794b1f04'
KIND='human_approved_single_GPU4_full115_retrieval_v1'


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def validate(c):
    item=c.get('approved_single_auxiliary')
    if item is None:return False
    intent=item['human_intent']
    if (item.get('kind')!=KIND or item['human_intent_sha256']!=digest(intent)
            or intent.get('human_text')!='可以的，继续'
            or not intent.get('retrieval_consistency_required')
            or intent.get('actor_indices')!=[0,1,2,3] or intent.get('auxiliary_indices')!=[4]
            or intent.get('auxiliary_uuid')!=GPU4
            or c.get('run_id')!='54ac33dcebe64e62a11fe4177f41f116'
            or not c.get('grpo_from_actor50') or not c.get('full_joint_fault_recovery')
            or c['resume_joint'].get('adapter')!='actor50_full_joint_v1'
            or c['resume_joint']['completed_outer']!=115
            or c['physical_gpu_indices']!=[0,1,2,3]
            or c['physical_gpu_uuids']!=intent['actor_uuids']
            or c['auxiliary_gpu_indices']!=[4] or c['auxiliary_gpu_uuids']!=[GPU4]
            or len(set(c['physical_gpu_uuids']+c['auxiliary_gpu_uuids']))!=5
            or c.get('retrieval_gpu_shards')!=1 or c.get('retrieval_logical_shards')!=2
            or c.get('retrieval_consistency_reference')!=str(Path(c['base'])/'preparation/retrieval-reference.json')
            or c.get('retrieval_consistency_questions')!=c['validation_questions']):
        raise ValueError('Single auxiliary GPU4 requires the explicit human full115 recovery binding')
    return True


def consistency_queries(questions_path):
    import pyarrow.parquet as pq
    rows=pq.read_table(questions_path).to_pylist()
    if len(rows)!=512:raise ValueError('Require the retained fixed512 question file')
    ordinals=list(range(0,512,8))
    queries=[rows[i]['prompt'][0]['content'] for i in ordinals]
    return queries,dict(question_file_sha256=hashlib.sha256(Path(questions_path).read_bytes()).hexdigest(),
        query_count=len(queries),query_ordinals=ordinals,query_sha256=digest(queries))


def check_retained_layout(retriever, reference, questions_path):
    """Match a separately generated original-layout reference, same64 queries."""
    import numpy as np
    queries,query_info=consistency_queries(questions_path)
    assert reference['kind']=='retained_original_two_shard_FP16_GPU4_layout_reference_v1'
    for key,value in query_info.items():
        if reference[key]!=value:raise ValueError('Retained retrieval query identity differs')
    embeddings=retriever.encoder.encode(queries)
    if hashlib.sha256(embeddings.tobytes()).hexdigest()!=reference['shared_query_embeddings_sha256']:
        raise ValueError('Retained retrieval encoder output differs')
    scores,ids=retriever.index.search(embeddings,3)
    if not np.isfinite(scores).all() or ids.shape!=(64,3):raise ValueError('Invalid original-layout retrieval')
    actual_hash=hashlib.sha256(ids.tobytes()).hexdigest()
    if actual_hash!=reference['ordered_document_ids_sha256']:
        raise ValueError('Original two-shard layout did not reproduce all64 ordered Top3 document IDs')
    if retriever.index.d!=reference['dimension'] or retriever.index.ntotal!=reference['documents']:
        raise ValueError('Retained retrieval index identity differs')
    return dict(status='PASS',**query_info,topk=3,exact_ordered_top3_matches=64,
        ordered_document_ids_sha256=actual_hash,reference_ordered_document_ids_sha256=reference['ordered_document_ids_sha256'],
        shared_query_embeddings_sha256=reference['shared_query_embeddings_sha256'],
        reference_source_execution=reference['source_execution_id'],reference_source_receipt_sha256=reference['source_receipt_sha256'],
        reference_physical_GPUs=1,reference_logical_shards=2,actual_physical_GPUs=1,actual_logical_shards=2,
        index_dimensions=int(retriever.index.d),documents=int(retriever.index.ntotal),
        encoder='unchanged_E5_FP16_mean_query_prefix',max_absolute_score_difference=None,
        score_difference_note='Only ordered IDs and exact input embeddings were retained from the separate reference',
        question_answers_or_QA_scoring_used=False,actor_or_probe_updates=0)
