"""Read only B128 evaluation progress and epoch01..04 complete receipts.

No checkpoint, model, features, predictions, scheduler or training writes.
Missing/transient/unknown metadata is reported without inventing metrics.
"""
import datetime,hashlib,json,time
from pathlib import Path
BASE=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/joint-mixed-direct-official-169028-b128-v1')


def read_small(path):
    try:
        if path.stat().st_size>1024*1024:return dict(state='UNEXPECTED_METADATA_OVER_1MiB',path=str(path)),None
        raw=path.read_bytes()
    except FileNotFoundError:return dict(state='MISSING',path=str(path)),None
    except OSError as exc:return dict(state='READ_UNAVAILABLE',path=str(path),error_type=type(exc).__name__,error=str(exc)),None
    ref=dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    try:value=json.loads(raw)
    except (ValueError,UnicodeError)as exc:return dict(state='JSON_UNAVAILABLE_OR_TRANSIENT',file=ref,error_type=type(exc).__name__,error=str(exc)),None
    if not isinstance(value,dict):return dict(state='UNRECOGNIZED_JSON_TYPE',file=ref,value=value),None
    return dict(state='PRESENT',file=ref),value


def metric_summary(value):
    if not isinstance(value,dict):return dict(state='MISSING'if value is None else'UNRECOGNIZED_TYPE',value=value)
    pool=value.get('pool');top=value.get('selected512');pool=pool if isinstance(pool,dict)else{};top=top if isinstance(top,dict)else{}
    fields=dict(binaryMixedBCE=value.get('binaryMixedBCE'),MixedBA=pool.get('mixed_balanced_accuracy'),Mixed_recall=pool.get('mixed_recall'),nonMixed_recall=pool.get('nonmixed_recall'),AUC=pool.get('mixed_roc_auc'))
    fields['missing_fields']=[k for k,v in fields.items()if v is None]
    fields.update(state='RECEIPT_METRICS',questions=pool.get('questions'),Top512=dict(selected_questions=top.get('questions'),actual_Mixed=top.get('mixed_questions'),Mixed_fraction=top.get('mixed_fraction'),selection_budget=value.get('selection_budget')))
    known={'pool','selected512','selection_budget','selected_qids_sha256','ordered_qids_sha256','ordered_own_trajectory_ids_sha256','labels_used_to_select','threshold_fitted','tie_break','score_semantics','trajectory_probability_or_accuracy_produced','product_formula_used','binaryMixedBCE','classwise_binaryMixedBCE','timing','predictions'}
    fields['unknown_metric_fields']={k:v for k,v in value.items()if k not in known}
    known_pool={'questions','mixed_questions','mixed_fraction','confusion','mixed_recall','nonmixed_recall','mixed_false_negative_rate','nonmixed_false_positive_rate','mixed_balanced_accuracy','mixed_roc_auc','classwise','k_histogram','mean_q','allwrong_false_positive_rate','allcorrect_false_positive_rate','Mixed_false_negative_rate','fixed_threshold'}
    fields['unknown_pool_fields']={k:v for k,v in pool.items()if k not in known_pool}
    fields['unknown_selected512_fields']={k:v for k,v in top.items()if k not in known_pool}
    return fields


def run():
    out=dict(observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),observed_epoch=time.time(),base=str(BASE),expected_training_job_id='746266',training_job_status_not_queried=True,model_feature_prediction_or_checkpoint_payload_read=False,GPU_used=False,new_scoring=False,training_files_changed=False)
    state,value=read_small(BASE/'evaluation-progress.json');out['evaluation_progress']=dict(state,reported=value)
    out['epochs']={}
    for number in range(1,5):
        state,value=read_small(BASE/'evaluations'/('epoch-%02d'%number)/'complete.json')
        if value is not None:
            state.update(reported_epoch=value.get('epoch'),cursor=value.get('cursor'),status=value.get('status'),checkpoint=value.get('checkpoint'),config_sha256=value.get('config_sha256'),reported_completed_epoch_time=value.get('completed_epoch_time'))
            metrics=value.get('metrics');metrics=metrics if isinstance(metrics,dict)else{}
            state['models']={name:{pool:metric_summary(metrics.get(name,{}).get(pool))if isinstance(metrics.get(name,{}),dict)else metric_summary(metrics[name])for pool in('diagnostic512','fixed_selection2048')}for name in('raw','EMA')}
            state['unknown_models']={k:v for k,v in metrics.items()if k not in('raw','EMA')}
            state['unknown_model_pools']={name:{k:v for k,v in values.items()if k not in('diagnostic512','fixed_selection2048')}for name,values in metrics.items()if name in('raw','EMA')and isinstance(values,dict)}
            known={'status','epoch','cursor','checkpoint','config_sha256','checkpoint_saved_before_evaluation','models_two_Adam_RNG_native_buffers_unchanged','metrics','validation_proof','final4096_scored','threshold_fitted','actor_updates','completed_epoch_time'}
            state['unknown_complete_fields']={k:v for k,v in value.items()if k not in known}
        out['epochs']['epoch-%02d'%number]=state
    return out


if __name__=='__main__':print(json.dumps(run(),sort_keys=True,allow_nan=False))
