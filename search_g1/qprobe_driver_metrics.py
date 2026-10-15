"""Optional scalar reporting; diagnostics must not invalidate an actor update."""
import math
from collections.abc import Mapping


def selection_diagnostics(audit):
    metrics={}
    def scalar(key,value):
        if isinstance(value,(int,float)) and not isinstance(value,bool):
            if math.isfinite(value):metrics[key]=value
            else:metrics['question_probe/diagnostic_reporting_incomplete']=1
    if not isinstance(audit,Mapping):
        return {'question_probe/diagnostic_reporting_incomplete':1}
    for name in ('selected_all_wrong','selected_all_correct','selected_random_exploration','selected_predicted_mixed'):
        if name in audit:scalar('selection/'+name,audit[name])
        else:metrics['question_probe/diagnostic_reporting_incomplete']=1
    prequential=audit.get('prequential')
    if not isinstance(prequential,Mapping):
        metrics['question_probe/diagnostic_reporting_incomplete']=1
        return metrics
    if prequential.get('diagnostics_complete') is False or 'diagnostic_error' in prequential:
        metrics['question_probe/diagnostic_reporting_incomplete']=1
    for slice_name,values in prequential.items():
        if not isinstance(values,Mapping) or slice_name=='diagnostic_error':continue
        for name,value in values.items():
            scalar('question_probe/prequential/'+str(slice_name)+'/'+str(name),value)
    return metrics
