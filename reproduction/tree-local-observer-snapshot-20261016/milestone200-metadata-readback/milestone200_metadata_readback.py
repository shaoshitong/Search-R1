#!/usr/bin/env python3
"""Future single-query original778 milestone200 JSON/stat preservation.

SOURCE PREPARATION ONLY; do not run until the actual original200 scalar/dev512
coverage product and its publication readback are frozen. This does not stop,
start, restore, evaluate or modify training. It performs one existing retained18
helper call, with two JSON/stat readbacks inside that call, and publishes a small
metadata receipt to BASE and both mirrors. There is no automatic remote retry.

Future CLI (every coverage pin comes from actual future local bytes):
  env TREE_MGMT_CONTROL_PATH=/tmp/tree-mgmt-3fw9a1ab/control python3 -B \
    work/tree-speed-repair-20261008/milestone200_metadata_readback.py \
    --coverage "$ACTUAL_VARIANCE200_COVERAGE" --coverage-bytes "$ACTUAL_BYTES" \
    --coverage-sha256 "$ACTUAL_SHA256"

Paths follow the already-published caac340 final HF contract: permanent native
checkpoints/training_state/global_step_200/manifest.json and HF actor/global_step_200.
Only ordinary native manifest/config/HF JSON metadata is read. PT and HF weight
payloads are stat-only; their hashes remain explicitly original-writer recorded.
Tokenizer/config/index JSON hashes are actually checked without loading a model
or tokenizer runtime. No tensor, GPU, raw rollout, cold reload or assets-to-Mac.
Saved scheduler200/cursor201 is a boundary, not actual training201 entry.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shlex
import time


W = Path(__file__).resolve().parents[2]
B = W / 'outputs/tree-nested-branch-handoff-20261013'
M = [W / 'work/tree-grpo/reproduction/baselines/tree-validation-20261002/tree-nested-branch',
     Path('/Users/shaoshitong/Documents/remote_codex_control/research/search-r1-baselines/results/tree-grpo/tree-nested-branch')]
HELPER = W / 'work/tree-speed-repair-20261008/ssh_retained18.py'
HELPER_SHA = '8065815f4452923bcb96fbdc12b96b246b7834f85f1bc2a9f75623912c157021'
C = '778b2b43dad90fc0c1c5d95d77c2326286011b73'
CONFIG = '355edf5232dc963deb526281f7e1e61b98eaaa73b0bffc5e7855ef3d262799b3'
MODEL = '744d9f616373b3890cf815412560e34041e21cd4915c026967d8cc13a423aea0'
DATA = '0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
RUN = '/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-nested-branch-fresh200-not-slurm;systemd:' + C
UNIT = 'sst-tree-nested-branch-fresh200-778b2b43.service'
INV = 'a313bda919834d24b4c5d491698d39ce'
OUTPUT = 'MILESTONE200_HF_NATIVE_METADATA.json'
READBACK = 'MILESTONE200_HF_NATIVE_METADATA_READBACKS.json'
COVERAGE_NAME = 'METRIC_RECORD_COVERAGE_THROUGH200_LOCAL.json'
COVERAGE_READBACK = 'METRIC_RECORD_COVERAGE_THROUGH200_LOCAL_READBACKS.json'
FIELDS = ['critic/rewards/mean', 'critic/score/mean',
          'tree_variance/within_root_raw_reward_population_variance',
          'tree_variance/intra_adv_population_variance', 'tree_variance/group_actual_adv_population_variance',
          'tree_variance/same_reward_question_fraction', 'tree_variance/mixed_reward_question_fraction',
          'tree_variance/exact_reward_one_question_count', 'tree_variance/exact_reward_zero_question_count',
          'tree_variance/root0_same_reward_fraction', 'tree_variance/root0_mixed_reward_fraction',
          'tree_variance/root1_same_reward_fraction', 'tree_variance/root1_mixed_reward_fraction',
          'tree_variance/same_reward_tree_fraction', 'tree_variance/mixed_reward_tree_fraction']
F, RAW = {}, {}

# Pure remote stdlib metadata code. EXPECTED is replaced only after all actual
# local gates pass. No writer, model import, shell command or tensor reader exists.
REMOTE = r'''
from pathlib import Path
import hashlib,json,os,socket,stat,time
expected=EXPECTED
assert socket.gethostname().split('.')[0]=='h100-10'
source=expected['source_commit'];run=Path(expected['run'])
native=run/'checkpoints/training_state/global_step_200/manifest.json'
hf=run/'checkpoints/actor/global_step_200';config=run/'resolved.yaml'
run_real=run.resolve(strict=True)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def digest(s):return isinstance(s,str) and len(s)==64 and all(c in '0123456789abcdef' for c in s)
def stat_file(path):
 path=Path(path);a=path.lstat();real=path.resolve(strict=True)
 assert stat.S_ISREG(a.st_mode) and a.st_size>0 and real.is_relative_to(run_real),str(path)
 b=path.stat();assert (a.st_dev,a.st_ino,a.st_size)==(b.st_dev,b.st_ino,b.st_size)
 return dict(path=str(path),real_path=str(real),bytes=a.st_size,mtime_ns=a.st_mtime_ns,ctime_ns=a.st_ctime_ns)
def read_metadata(path,max_bytes=16777216):
 before=stat_file(path);assert before['bytes']<=max_bytes,str(path)
 raw=Path(path).read_bytes();assert stat_file(path)==before
 return raw,before
def strict(raw):
 def unique(pairs):
  out={}
  for k,v in pairs:
   assert k not in out,'duplicate JSON key: '+k
   out[k]=v
  return out
 return json.loads(raw,object_pairs_hook=unique,parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
def identity(value):
 for key,val in dict(code_commit=source,config_sha256=expected['config_sha256'],
       model_identity=expected['model_identity'],data_identity=expected['data_identity'],
       job_id='not-slurm;systemd:'+source,completed_step=200,actual_update_limit=200).items():
  assert value[key]==val,key
def observe():
 raw,native_stat=read_metadata(native,262144);d=strict(raw)
 assert d['status']=='COMPLETE_SAME_WORLD_STATE_SAVED'
 identity(d['identity'])
 accounting=d['identity']['guard_continue_accounting'];totals=accounting['total_optimizer_calls_per_rank']
 assert totals==expected['actual_total_adam_per_rank'] and set(totals)=={str(i) for i in range(8)}
 assert all(type(x)is int and x>=0 for x in totals.values())
 assert accounting['completed_outer_updates']==200
 assert accounting['completed_optimizer_calls_per_rank']==totals
 assert not any(accounting['partial_optimizer_calls_per_rank'].values())
 workers=d['worker_states'];assert len(workers)==8 and {r['identity']['rank'] for r in workers}==set(range(8))
 declared={str(native),str(native.parent/'driver.pt')}
 for r in workers:
  identity(r['identity']);rank=r['identity']['rank'];calls=totals[str(rank)]
  assert type(rank)is int and r['identity']['world_size']==8
  assert Path(r['path'])==native.parent/('rank-%02d.pt'%rank)
  assert r['actual_total_optimizer_calls']==calls and r['observed_adam_steps']==[calls]
  assert r['scheduler_last_epoch']==200 and r['full_worker_state_finite'] is True
  assert digest(r['sha256']) and digest(r['logical_state_sha256'])
  observed=stat_file(r['path']);assert observed['bytes']==r['bytes']
  declared.add(r['path'])
 assert Path(d['driver']['path'])==native.parent/'driver.pt' and d['driver']['next_step']==201
 assert digest(d['driver']['sha256']) and digest(d['driver']['logical_state_sha256'])
 observed_native={str(p) for p in native.parent.iterdir() if p.is_file()}
 assert observed_native==declared
 native_files=[stat_file(p) for p in sorted(declared)]
 assert Path(d['hf_checkpoint'])==hf
 files=d['hf_identity']['files'];assert isinstance(files,dict) and files
 assert all(isinstance(name,str) and name and not Path(name).is_absolute() and '..' not in Path(name).parts
            and digest(recorded) for name,recorded in files.items())
 assert d['hf_identity']['sha256']==sha(json.dumps(files,sort_keys=True).encode())
 inventory={str(p.relative_to(hf)) for p in hf.rglob('*') if p.is_file()}
 assert inventory==set(files)
 hf_files=[dict(stat_file(hf/name),name=name,recorded_sha256=recorded,
                recorded_hash_scope='original writer recorded; actual JSON hashes are listed separately')
           for name,recorded in sorted(files.items())]
 weights={name for name in files if name.endswith('.safetensors') or (Path(name).name.startswith('pytorch_model') and name.endswith('.bin'))}
 required={'config.json','tokenizer_config.json','tokenizer.json','special_tokens_map.json',
           'added_tokens.json','vocab.json','merges.txt','generation_config.json'}
 assert required<=set(files) and weights
 metadata={};json_values={}
 for name in sorted(files):
  if name.endswith('.json'):
   content,record=read_metadata(hf/name);assert sha(content)==files[name]
   value=strict(content);assert isinstance(value,dict)
   metadata[name]=dict(record,sha256=sha(content),actual_metadata_json_sha256_recomputed=True)
   json_values[name]=value
 indexes={name:value for name,value in json_values.items() if name.endswith('.index.json')}
 assert indexes
 indexed=set()
 for name,value in indexes.items():
  wm=value['weight_map'];assert isinstance(wm,dict) and wm
  assert all(isinstance(k,str) and isinstance(v,str) and v in weights for k,v in wm.items())
  indexed.update(wm.values())
 assert indexed==weights
 model=json_values['config.json'];tok=json_values['tokenizer_config.json'];backend=json_values['tokenizer.json']
 assert model.get('model_type')=='qwen2' and model.get('architectures')==['Qwen2ForCausalLM']
 assert type(model.get('vocab_size'))is int and model['vocab_size']>0
 assert isinstance(tok.get('tokenizer_class'),str) and isinstance(tok.get('chat_template'),str)
 assert isinstance(backend.get('model'),dict) and isinstance(backend['model'].get('vocab'),dict)
 raw_config,config_stat=read_metadata(config,1048576);assert sha(raw_config)==expected['config_sha256']
 assert native.read_bytes()==raw
 return dict(native_manifest_path=str(native),native_manifest_sha256=sha(raw),native_manifest=d,
    native_manifest_file=native_stat,native_files=native_files,hf_directory=str(hf),hf_files=hf_files,
    hf_identity=d['hf_identity'],actual_hf_metadata_json=metadata,
    resolved_training_config=dict(config_stat,sha256=sha(raw_config)),
    tokenizer_config_summary=dict(tokenizer_class=tok['tokenizer_class'],model_max_length=tok.get('model_max_length'),
       chat_template_sha256=sha(tok['chat_template'].encode()),backend_model_type=backend['model'].get('type'),
       backend_vocab_entries=len(backend['model']['vocab']),special_token_role_keys=sorted(json_values['special_tokens_map.json'])),
    hf_config_summary={k:model.get(k) for k in ('model_type','architectures','vocab_size','hidden_size','num_hidden_layers','bos_token_id','eos_token_id','torch_dtype')},
    shard_index_files=sorted(indexes),weight_shard_names=sorted(weights),
    actual_total_adam_per_rank=totals,saved_scheduler_last_epoch=200,saved_driver_next_step=201)
first=observe();first_epoch=time.time();time.sleep(1);second=observe();second_epoch=time.time()
assert first==second,'metadata or stat changed across readbacks'
out=dict(status='ACTUAL_MILESTONE200_HF_NATIVE_METADATA_OBSERVED',source_commit=source,run_directory=str(run),
    observed_epoch=second_epoch,**second,
    metadata_readback_observations=[dict(pass_number=1,epoch=first_epoch),dict(pass_number=2,epoch=second_epoch)],
    tensor_contents_read=False,HF_weight_payloads_read=False,recorded_tensor_or_HF_weight_sha256_recomputed=False,
    independent_cold_reload_performed=False,tokenizer_runtime_loaded=False,GPU_used=False,
    training_signals_sent=False,frozen_source_modified=False,remote_files_written=False,
    saved_cursor201_is_not_actual_training201_entry=True,
    limits='Actual permanent native200 COMPLETE JSON, original writer finite/Adam/scheduler/HF identity, positive stable same-run file stat and actual small metadata JSON hashes only. Native PT and HF weight payloads/hashes are not read/recomputed. No model/tokenizer cold reload, tensor-finite recomputation, native-HF parameter equivalence, current/lifetime availability after capture, or final51713 acceptance is claimed.')
print(json.dumps(out,separators=(',',':'),ensure_ascii=False,allow_nan=False))
'''


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def local(value):
    p = Path(value)
    assert '..' not in p.parts and not p.is_symlink()
    if not p.is_absolute():
        p = W / p if p.parts and p.parts[0] == 'outputs' else B / p
    p = p.resolve(strict=True)
    assert p.is_relative_to(B) and p.suffix == '.json'
    return p


def read(value, size=None, expected=None, mirrored=False):
    p = local(value);raw = p.read_bytes()
    assert 0 < len(raw) <= 4 * 1024 * 1024 and p.is_file() and not p.is_symlink()
    assert size is None or type(size) is int and len(raw) == size
    assert expected is None or digest(expected) and sha(raw) == expected
    if str(p) in RAW:
        assert RAW[str(p)] == raw and F[str(p)]['mirrored'] == mirrored
    if mirrored:
        for mirror in M:
            assert (mirror / p.relative_to(B)).read_bytes() == raw
    F[str(p)], RAW[str(p)] = dict(path=str(p.relative_to(W)), bytes=len(raw), sha256=sha(raw), mirrored=mirrored), raw
    def unique(pairs):
        out = {}
        for k, v in pairs:
            assert k not in out
            out[k] = v
        return out
    value = json.loads(raw, object_pairs_hook=unique, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    assert isinstance(value, dict)
    return value


def bound(ref, mirrored=False):
    assert type(ref['bytes']) is int and ref['bytes'] > 0 and digest(ref['sha256'])
    return read(ref['path'], ref['bytes'], ref['sha256'], mirrored)


def ref(value):
    return {k: v for k, v in F[str(local(value))].items() if k != 'mirrored'}


def unchanged():
    for name, record in F.items():
        p = Path(name);raw = p.read_bytes()
        assert raw == RAW[name] and len(raw) == record['bytes'] and sha(raw) == record['sha256']
        if record['mirrored']:
            for mirror in M:
                assert (mirror / p.relative_to(B)).read_bytes() == raw


def encode(value):
    return (json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n').encode()


def main():
    assert __debug__, 'actual conditions cannot be disabled with python -O'
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--coverage', required=True, type=Path)
    p.add_argument('--coverage-bytes', required=True, type=int)
    p.add_argument('--coverage-sha256', required=True)
    args = p.parse_args()
    assert args.coverage_bytes > 0 and digest(args.coverage_sha256)
    assert os.environ.get('TREE_MGMT_CONTROL_PATH') == '/tmp/tree-mgmt-3fw9a1ab/control'
    assert not any((folder / name).exists() for folder in [B] + M for name in (OUTPUT, READBACK))
    assert local(args.coverage) == B / COVERAGE_NAME
    coverage = read(str(args.coverage), args.coverage_bytes, args.coverage_sha256, True)
    assert coverage['schema'] == 'tree_nested_metric_record_coverage_through200_local_v1'
    assert coverage['status'] == 'PASS_ACTUAL_ORIGINAL_SCALAR_COVERAGE_THROUGH200' and coverage['required_field_gaps'] == {}
    assert coverage['derived_scalar_replacements'] == [] and len(coverage['input_readback_observations']) == 2
    by_field = {v['original_field']: v for v in coverage['coverage'].values()}
    assert set(by_field) == set(FIELDS)
    for field, series in by_field.items():
        assert series['actual_covered_outers'] == list(range(1, 201)) and series['unknown_outers'] == []
        assert len(series['values_in_outer1_to200_order']) == 200 and all(numeric(v) for v in series['values_in_outer1_to200_order'])
    publication = read(COVERAGE_READBACK, mirrored=True)
    assert publication['schema'] == 'variance200_record_coverage_publication_readback_v1'
    assert publication['product']['bytes'] == args.coverage_bytes and publication['product']['sha256'] == args.coverage_sha256
    assert len(publication['observations']) == 2 and all(x['product_copies'] == 3 and x['source_bytes_and_mirrors_preserved'] is True for x in publication['observations'])
    gate = coverage['actual200_boundary']
    assert gate['saved_cursor_is_not_actual201_entry'] is True and gate['native_saved_scheduler_last_epoch'] == 200 and gate['native_saved_driver_next_step'] == 201
    totals = gate['actual_total_adam_per_rank']
    assert set(totals) == {str(i) for i in range(8)} and all(type(v) is int and v >= 0 for v in totals.values())
    control = bound(gate['dev200_periodic_control'], True)
    poll = bound(gate['progress_poll'])
    native = bound(gate['native_metadata'], True)
    assert poll['source_commit'] == C and poll['unit'] == UNIT and poll['run'] == RUN
    assert poll['progress']['actual_completed_outer_from_attempt_results'] == 200 and not poll['progress']['errors']
    metrics = [v for v in poll['iteration_metrics'] if v.get('step') == 200]
    assert metrics and all(v == metrics[0] for v in metrics)
    assert all(metrics[0][field] == by_field[field]['values_in_outer1_to200_order'][-1] for field in FIELDS)
    assert native['outer'] == 200 and native['source_commit'] == C and native['identity']['completed_step'] == 200
    assert {str(w['rank']):w['actual_total_optimizer_calls'] for w in native['receipt']['workers']} == totals
    assert control['status'] == 'VALIDATION_COST_COMPLETE' and control['phase'] == 'periodic'
    assert control['actual_completed_outer'] == 200 and control['evaluated_rows'] == control['dataset_rows'] == 512
    assert control['ledger_sha256'] == gate['original_dev200_inner_ledger_sha256']
    for key, value in dict(code_commit=C, config_sha256=CONFIG, model_identity=MODEL, data_identity=DATA).items():
        assert control[key] == value
    assert control['metrics']['val/qa_count/nq'] == control['metrics']['val/qa_count/hotpotqa'] == 256
    helper_raw = HELPER.read_bytes()
    assert len(helper_raw) == 4411 and sha(helper_raw) == HELPER_SHA
    expected = dict(source_commit=C, config_sha256=CONFIG, model_identity=MODEL, data_identity=DATA,
                    run=RUN, actual_total_adam_per_rank=totals)
    assert REMOTE.count('EXPECTED') == 1
    query = REMOTE.replace('EXPECTED', repr(expected));query_sha = sha(query.encode())
    unchanged()
    spec = importlib.util.spec_from_file_location('milestone200_retained18_transport', HELPER)
    helper = importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    result = helper.run('python3 -B -c ' + shlex.quote(query), target='h100-10', timeout=60)
    assert result.returncode == 0, 'one metadata query failed; no automatic retry: ' + result.stderr[-2000:]
    assert 0 < len(result.stdout.encode()) <= 128 * 1024
    observed = json.loads(result.stdout)
    assert observed['status'] == 'ACTUAL_MILESTONE200_HF_NATIVE_METADATA_OBSERVED'
    assert observed['source_commit'] == C and observed['run_directory'] == RUN
    assert observed['actual_total_adam_per_rank'] == totals and len(observed['metadata_readback_observations']) == 2
    value = dict(observed, local_received_at_utc=datetime.now(timezone.utc).isoformat(),
                 original200_scalar_dev_gate=ref(str(args.coverage)), gate_publication_readback=ref(COVERAGE_READBACK),
                 gate_native_metadata=gate['native_metadata'], gate_original200_periodic_control=gate['dev200_periodic_control'],
                 actual_training_invocation_id=INV, actual_training_pid=503255,
                 metadata_query_provenance=dict(transport_path=str(HELPER.relative_to(W)), transport_sha256=HELPER_SHA,
                     target='h100-10', query_sha256=query_sha, query=query, remote_exit_code=0,
                     actual_query_invocations_this_task=1, automatic_retry=False),
                 generator=dict(path=str(Path(__file__).resolve()), bytes=Path(__file__).stat().st_size,
                                sha256=sha(Path(__file__).read_bytes())),
                 local_input_pins=sorted(F.values(), key=lambda r:r['path']),
                 before_authorized_stop_metadata_only=True, final51713_or_HF_cold_reload_acceptance_claimed=False)
    raw = encode(value);assert len(raw) <= 128 * 1024
    targets = [folder / OUTPUT for folder in [B] + M]
    assert not any((folder / name).exists() for folder in [B] + M for name in (OUTPUT, READBACK))
    for target in targets:
        with target.open('xb') as stream:
            stream.write(raw)
    observations = []
    for n in range(2):
        unchanged();assert HELPER.read_bytes() == helper_raw
        for target in targets:
            assert target.read_bytes() == raw
        observations.append(dict(pass_number=n + 1, observed_utc=datetime.now(timezone.utc).isoformat(), copies=3,
                                 local_gate_bytes_preserved=True, actual_remote_query_count=1))
        if n == 0:
            time.sleep(1)
    receipt = encode(dict(schema='milestone200_metadata_publication_readback_v1',
                          product=dict(path=str(targets[0]), bytes=len(raw), sha256=sha(raw)),
                          observations=observations, weights_or_PT_contents_read=False, training_modified=False))
    for folder in [B] + M:
        with (folder / READBACK).open('xb') as stream:
            stream.write(receipt)
    for folder in [B] + M:
        assert (folder / READBACK).read_bytes() == receipt
    print(json.dumps(dict(status=value['status'], path=str(targets[0]), bytes=len(raw), sha256=sha(raw),
                          readback_bytes=len(receipt), readback_sha256=sha(receipt))))


if __name__ == '__main__':
    main()
