"""Pure CPU mechanics of exact-input historical G1 reward reconstruction.

No Torch import, no model load, no writes, no raw records exported. Historical
online row rewards were not persisted; this module never names a reconstructed
value an original online measurement.
"""
import ast
import hashlib
import heapq
import json
import math
from pathlib import Path
import struct
from types import SimpleNamespace
from typing import List, Tuple


class Hold(ValueError):
    pass


def require(value, reason):
    if not value:
        raise Hold(reason)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def extract_functions(path, names, namespace, class_name=None):
    """Compile exact AST bodies without importing the production server/trainer."""
    module = ast.parse(Path(path).read_text())
    pool = module.body
    if class_name:
        classes = [x for x in pool if isinstance(x, ast.ClassDef) and x.name == class_name]
        require(len(classes) == 1, 'native_class_not_unique')
        pool = classes[0].body
    selected = [x for x in pool if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef))
                and x.name in names]
    require({x.name for x in selected} == set(names), 'native_function_missing')
    selected.sort(key=lambda x: names.index(x.name))
    code = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(code)
    exec(compile(code, str(path), 'exec'), namespace)
    return {name: namespace[name] for name in names}


def load_balance(path, expected_sha):
    require(sha_file(path) == expected_sha, 'native_balance_source_changed')
    ns = dict(heapq=heapq, List=List, Tuple=Tuple)
    funcs = extract_functions(path, ['karmarkar_karp', 'get_seqlen_balanced_partitions'], ns)
    return funcs['get_seqlen_balanced_partitions']


def balance_order(lengths, fn, world=4):
    require(world == 4 and len(lengths) % world == 0 and
            all(type(x) is int and x > 0 for x in lengths), 'invalid_native_balance_inputs')
    parts = fn(lengths, k_partitions=world, equal_size=True)
    indices = [j for partition in parts for j in partition]
    require(sorted(indices) == list(range(len(lengths))) and
            all(len(p) * world == len(lengths) for p in parts), 'balance_not_equal_permutation')
    return indices


def verify_balance_method(path, expected_sha):
    require(sha_file(path) == expected_sha, 'native_trainer_source_changed')
    tree = ast.parse(Path(path).read_text())
    methods = [x for x in ast.walk(tree) if isinstance(x, ast.FunctionDef) and x.name == '_balance_batch']
    require(len(methods) == 1, 'native_balance_method_not_unique')
    code = ast.unparse(methods[0])
    required = ["batch.batch['attention_mask']", '.sum(-1).tolist()',
                'self.actor_rollout_wg.world_size', 'equal_size=True',
                'for partition in global_partition_lst for j in partition', 'batch.reorder(global_idx)']
    require(all(x in code for x in required), 'unsupported_native_balance_method')
    return hashlib.sha256(ast.dump(methods[0], include_attributes=False).encode()).hexdigest()


def load_reward(path, expected_sha, constants):
    require(sha_file(path) == expected_sha, 'native_reward_source_changed')
    require(constants == dict(PENALTY=0.1, MAX_SEARCH=2, wrong_floor=0.05,
            bal_ground=0.6, bal_nosearch_bonus=0.5, w_cost=0.02,
            wrong_ground_scale=0.0, probe_base=0.0, probe_gate=True,
            probe_alpha_cost=False), 'unsupported_or_unbound_reward_parameters')
    tree = ast.parse(Path(path).read_text())
    classes = [x.name for x in tree.body if isinstance(x, ast.ClassDef) and
               any(isinstance(f, ast.FunctionDef) and f.name == '_score_probe' for f in x.body)]
    require(len(classes) == 1, 'native_reward_owner_not_unique')
    fn = extract_functions(path, ['_score_probe'], {}, classes[0])['_score_probe']
    instance = SimpleNamespace(**constants)
    return lambda row, pr: fn(instance, row, pr)[0]


def scalar_reward(score_fn, valid, correct, n_search, b, d):
    require(type(valid) is bool and type(correct) is int and correct in (0, 1) and
            (not correct or valid), 'invalid_outcome')
    require(type(n_search) is int and n_search >= 0, 'invalid_authored_search_count')
    require(all(math.isfinite(x) and 0 <= x <= 1 for x in (b, d)), 'nonfinite_probe_probability')
    value = score_fn(dict(answer='VALID' if valid else None, correct=bool(correct),
                         n_search=n_search), dict(boundary=float(b), flip=float(d)))
    require(math.isfinite(value), 'nonfinite_reward')
    # The native tensor is float32; monitoring's sum_reward uses Python float.
    fp32 = struct.unpack('<f', struct.pack('<f', value))[0]
    return dict(R_reconstructed=fp32, R_python_before_FP32=value)


def request_items(actor_rows, c_rows, gate, c_helper, tokenizer, signals):
    """Stream original actor rows, retain decoded strings only, verify C IDs/hashes.

    Caller authenticates whole trace and each C artifact before invoking this.
    Decode/mask/strict outcome comes from the separately published C V2 helper.
    No raw token arrays are retained across iterations.
    """
    requests, lengths, digest = [], [], hashlib.sha256(b'[')
    for i, row in enumerate(actor_rows):
        require(i < len(c_rows), 'more_actor_than_C_rows')
        c_helper.verify_streamed_actor(row, i, gate)
        if i:
            digest.update(b',')
        digest.update(canonical(row))
        c = c_rows[i]
        require(c['trajectory_id'] == row['trajectory_id'], 'C_actor_ID_mismatch')
        prompt_ids, response_ids, model_mask = c_helper.actor_parts(row)
        prompt = tokenizer.decode(prompt_ids)
        response, ranges = signals.decode_segments(tokenizer, response_ids, model_mask)
        ctx = signals.contexts(prompt, response, ranges)
        valid = ctx is not None and ctx['answer'] is not None
        count = signals.model_search_count(response, ranges)
        for field, actual in [('decoded_prompt_sha256', hashlib.sha256(prompt.encode()).hexdigest()),
                              ('decoded_response_sha256', hashlib.sha256(response.encode()).hexdigest()),
                              ('model_ranges_sha256', hashlib.sha256(canonical(ranges)).hexdigest())]:
            require(c[field] == actual, 'C_decode_or_mask_projection_mismatch')
        require(valid == c['final_action_valid'] and count == c['model_authored_search_count'],
                'C_outcome_or_search_count_mismatch')
        requests.append(dict(id=row['trajectory_id'], prompt=prompt, response=response,
                             model_ranges=ranges, valid_rlen=len(response_ids), C_record=c))
        lengths.append(sum(row['attention_mask']))
    digest.update(b']')
    c_helper.verify_actor_array_digest(digest.hexdigest(), len(requests), gate)
    require(len(requests) == len(c_rows) == 2560, 'incomplete_replay_requests')
    return requests, lengths, digest.hexdigest()


def compact_sufficient_statistics(rows):
    """Exact individual pairs only; no batch/calibration surrogate Pearson."""
    require(rows and len({r['trajectory_id'] for r in rows}) == len(rows), 'duplicate_or_empty_pair_rows')
    n = len(rows)
    c = math.fsum(r['C_raw'] for r in rows)
    r = math.fsum(x['R_reconstructed'] for x in rows)
    r2 = math.fsum(x['R_reconstructed'] ** 2 for x in rows)
    cr = math.fsum(x['C_raw'] * x['R_reconstructed'] for x in rows)
    numerator = cr - c * r / n
    denominator = math.sqrt((c - c * c / n) * (r2 - r * r / n))
    return dict(N=n, sum_C=c, sum_R=r, sum_R2=r2, sum_CR=cr,
                individual_pair_Pearson=None if denominator == 0 else numerator / denominator,
                Pearson_semantics='strict C versus independent reconstructed shaped R, every row in this step',
                original_online_row_rewards_available=False)
