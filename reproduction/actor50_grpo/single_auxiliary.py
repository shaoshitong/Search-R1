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
            or c.get('retrieval_gpu_shards')!=1
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


def check_retrieval(retriever, reference_scores, reference_ids, embeddings, query_info):
    """Original two-shard layout and single Flat search share exact embeddings.

    Both reference shards ran sequentially before the single index conversion,
    on the one approved physical GPU. This is not a two-physical-GPU test.
    """
    import numpy as np
    actual_scores,actual_ids=retriever.index.search(embeddings,3)
    if not np.isfinite(actual_scores).all() or not np.isfinite(reference_scores).all():
        raise ValueError('Nonfinite retrieval consistency score')
    matches=np.all(actual_ids==reference_ids,axis=1)
    result=dict(status='PASS' if bool(matches.all()) else 'FAIL',**query_info,topk=3,
        exact_ordered_top3_matches=int(matches.sum()),ordered_document_ids_sha256=hashlib.sha256(actual_ids.tobytes()).hexdigest(),
        reference_ordered_document_ids_sha256=hashlib.sha256(reference_ids.tobytes()).hexdigest(),
        shared_query_embeddings_sha256=hashlib.sha256(embeddings.tobytes()).hexdigest(),
        max_absolute_score_difference=float(np.max(np.abs(actual_scores-reference_scores))),
        reference='original_FAISS_FP16_two_shard_cloner_both_on_approved_GPU4',
        reference_physical_GPUs=1,reference_logical_shards=2,actual_physical_GPUs=1,actual_logical_shards=1,
        layouts_evaluated_sequentially=True,index_copy_block_vectors=65536,
        encoder='unchanged_E5_FP16_mean_query_prefix',index_dimensions=int(retriever.index.d),documents=int(retriever.index.ntotal),
        question_answers_or_QA_scoring_used=False,actor_or_probe_updates=0)
    if not matches.all():raise ValueError('Single GPU top3 differs from original two-shard layout: '+json.dumps(result))
    return result
