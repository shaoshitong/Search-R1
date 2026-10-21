"""Owned-allocation diagnostic of actual retained-row decode; never trains.

This reads only bounded selected manifest/index metadata plus actual selected
rows. It does not repeat all276GB inventorySHA and does not qualify a training
reader. Every selected decoded row still passes its authentic retainedSHA.
"""
import argparse, cProfile, hashlib, io, json, os, pstats, random, time
from collections import OrderedDict
from pathlib import Path
import torch
from . import probe_ablation_train as u
from .replay_feature_reader import ReplayFeatureReader
from .replay_feature_reader_fast import FastReplayFeatureReader


def diagnostic_reader(c):
    reader = ReplayFeatureReader.__new__(ReplayFeatureReader)
    reader.layer, reader.dimension = 26, 3584
    reader.records, reader.cache, reader.prototypes = {}, OrderedDict(), {}
    reader.synthetic_fixture, reader.closed = False, False
    descriptors = u.read(u.checked(c['features']))
    selected = []
    used_refs = []
    # A bounded same-file sample deliberately isolates decode overhead from
    # random shared-storage latency. No scan through all1.69M row metadata.
    for ref in descriptors:
        manifest = u.read(u.checked(ref))
        index = Path(ref['path']).parent / 'index.jsonl'
        u.checked(manifest['files']['index.jsonl'])
        used_refs.append(ref)
        with index.open() as stream:
            for line in stream:
                row = json.loads(line)
                reader.validate_row(row, manifest)
                tid = row['trajectory_id']
                reader.records[tid] = (str(index.parent), row)
                selected.append(tid)
                if len(selected) == 320:
                    return reader, [selected[i:i+10] for i in range(0,320,10)], used_refs
    raise ValueError('Insufficient bounded actual diagnostic rows')


def run(source_config, output):
    if not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Actual reader profile requires own compute allocation')
    started = time.time()
    c = u.read(source_config)
    base, sample, manifests = diagnostic_reader(c)
    fast = FastReplayFeatureReader.__new__(FastReplayFeatureReader)
    fast.__dict__.update(base.__dict__)
    # Independent bounded file/cache state, same own sealed records.
    fast.cache, fast.prototypes = OrderedDict(), {}
    result = dict(status='ACTUAL_READER_DIAGNOSTIC_IN_PROGRESS', job_id=os.environ['SLURM_JOB_ID'],
        source_config_sha256=u.sha(source_config), microbatches=32, real_rows=320, manifests=manifests, sample_locality='first320 consecutive verified real rows; isolates decode overhead; not a random training batch',
        full_inventory_SHA_repeated=False, all_selected_retained_row_SHA_mandatory=True,
        benchmark_only_not_training_acceptance=True, cases=[])
    u.atomic(output, result)
    expected = []
    for threads, kind, reader in ((8, 'legacy', base), (1, 'legacy', base), (8, 'numpy', fast), (1, 'numpy', fast)):
        torch.set_num_threads(threads)
        durations = []
        profile = cProfile.Profile()
        for rep in range(2):
            elapsed = 0.
            for number, group in enumerate(sample):
                profile.enable()
                start = time.perf_counter()
                batch = reader.get(group, torch.device('cpu'))
                elapsed += time.perf_counter() - start
                profile.disable()
                if kind == 'legacy' and threads == 8 and rep == 0:
                    #32 small hashes remain on the remote compute allocation only.
                    expected.append(u.tensor_hash({k: batch[k] for k in ('features','attention_mask','position_ids')}))
                else:
                    actual = u.tensor_hash({k: batch[k] for k in ('features','attention_mask','position_ids')})
                    if actual != expected[number]:
                        raise ValueError('Optimized diagnostic reader changed actual retained tensor bits')
            durations.append(elapsed)
        stream = io.StringIO()
        pstats.Stats(profile,stream=stream).strip_dirs().sort_stats('cumulative').print_stats(18)
        case = dict(kind=kind, torch_CPU_threads=threads, seconds_per_32_questions=durations,
            input_tensor_bits_masks_positions_equal=True, cprofile=stream.getvalue())
        result['cases'].append(case)
        u.atomic(output, result)
    torch.set_num_threads(1)
    batch = fast.get(sample[0], torch.device('cpu'))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        t=time.perf_counter();device_batch={k:v.cuda() if torch.is_tensor(v) else v for k,v in batch.items()};torch.cuda.synchronize()
        result['one_question_10rows_host_to_device_seconds'] = time.perf_counter()-t
        result['GPU'] = torch.cuda.get_device_name()
        del device_batch
    base.close();fast.close()
    result.update(status='ACTUAL_READER_TIMING_AND_BIT_EXACT_COMPARISON_PASSED', elapsed_seconds=time.time()-started)
    u.atomic(output,result)
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-config',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    run(a.source_config,a.output)
