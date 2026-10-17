"""CPU-only same-effective-input label conflict counts; never reads hidden tensors.

The accuracy ceiling is conditional on treating identical token/context/mask/
position inputs as the same deterministic classifier input. It does not claim
bitwise identity of BF16 hidden captured with different execution batch shapes.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def inspect(scan, outcome):
    import numpy as np
    from reproduction.prefix_selection200.probe import prefix_row
    scan=Path(scan);proof=json.loads((scan/'scan-complete.json').read_text())
    arrays=Path(proof.get('source_inputs_dir',str(scan)))
    contexts=np.load(arrays/'packed_context_ids.npy',mmap_mode='r')
    labels={r['trajectory_id']:r['strict_em']for r in map(json.loads,Path(outcome).read_text().splitlines())}
    classes=defaultdict(Counter);questions=defaultdict(list);count=0
    for original in map(json.loads,(scan/'index.jsonl').read_text().splitlines()):
        row=prefix_row(original);label=labels[row['trajectory_id']]
        h=hashlib.sha256();qlen=row['question_length'];off=row['question_context_offset']
        h.update(json.dumps(dict(question_length=qlen,policy_length=row['policy_length'])).encode())
        h.update(contexts[off:off+qlen].tobytes())
        for segment in row['segments']:
            signature={k:segment[k]for k in ('request_length','context_length','policy_start','policy_length','position_start')}
            h.update(json.dumps(signature,sort_keys=True).encode())
            lo=segment['context_offset'];h.update(contexts[lo:lo+segment['context_length']].tobytes())
        key=h.hexdigest();classes[key][label]+=1;questions[str(row['qid'])].append(key);count+=1
    assert count==len(labels) and all(len(v)==5 for v in questions.values())
    conflict=[v for v in classes.values()if v[0] and v[1]]
    unavoidable=sum(min(v[0],v[1])for v in classes.values())
    import math
    bce_floor=sum(-sum(n*math.log(n/sum(v.values()))for n in v.values()if n)for v in classes.values())/count
    return dict(trajectories=count,questions=len(questions),unique_effective_inputs=len(classes),
        repeated_input_classes=sum(sum(v.values())>1 for v in classes.values()),
        conflicting_input_classes=len(conflict),trajectories_in_conflicting_classes=sum(sum(v.values())for v in conflict),
        minimum_mistakes_for_deterministic_identical_input_classifier=unavoidable,
        empirical_trajectory_accuracy_ceiling=1-unavoidable/count,
        empirical_trajectory_bce_floor=bce_floor,
        within_question_distinct_effective_input_histogram=dict(Counter(len(set(v))for v in questions.values())),
        mask_and_position_semantics='fixedQ273+policy20; authentic full causal contexts and segment positions included',
        theoretical_input_equivalence_not_bitwise_bf16_hidden_identity=True,
        hidden_tensor_loaded=False,model_forward_performed=False,training_modified=False)
