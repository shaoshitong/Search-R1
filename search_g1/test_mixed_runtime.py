"""Verify that selection preserves real sampling values and trajectory metadata."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import numpy as np
import torch
from verl import DataProto
from search_g1.mixed_runtime import subset, padded_concat, ROW_META
from search_g1 import mixed_runtime as runtime


def chunk(offset, width):
    rows = 10
    tensors = {'prompts': torch.full((rows, 3), 9, dtype=torch.long),
               'responses': torch.arange(rows*width).reshape(rows,width)+offset,
               'input_ids': torch.full((rows,3+width),9,dtype=torch.long),
               'sampling_log_probs': torch.arange(rows*width).reshape(rows,width).float()/100,
               'sampling_mask': torch.ones(rows,width,dtype=torch.bool)}
    return DataProto.from_dict(tensors=tensors,
        non_tensors={'trajectory_id': np.array([str(offset+i) for i in range(rows)],dtype=object)},
        meta_info={k:list(range(offset,offset+rows)) for k in ROW_META})


class RuntimeSelectionTests(unittest.TestCase):
    def test_selection_from_multiple_candidate_chunks(self):
        first, second = chunk(0,2),chunk(100,4)
        picked = [subset(first,[5,6,7,8,9]),subset(second,[0,1,2,3,4])]
        original_probs = picked[0].batch['sampling_log_probs'].clone()
        merged = padded_concat(picked,0)
        self.assertEqual(len(merged),10)
        for key in ROW_META:self.assertEqual(merged.meta_info[key],[5,6,7,8,9,100,101,102,103,104])
        self.assertEqual(merged.non_tensor_batch['trajectory_id'].tolist(),['5','6','7','8','9','100','101','102','103','104'])
        self.assertTrue(torch.equal(merged.batch['sampling_log_probs'][:5,:2],original_probs))
        self.assertFalse(bool(merged.batch['sampling_mask'][:5,2:].any()))
        self.assertEqual(int(merged.batch['sampling_mask'].sum()),30)

    def test_bad_row_metadata_rejected(self):
        data=chunk(0,2);data.meta_info['turns_stats']=[1]
        with self.assertRaises(ValueError):subset(data,[0])


class FixedPoolStream:
    """CPU source fixture; actual selector and collection control flow run intact."""
    def __init__(self):
        self.consumed = 0; self.pending = None; self.offset = 0
        self.pools = []; self.materializations = []
        self.binding = dict(stream_kind='cpu-fixed-eligible-fixture', eligible_pool_sha256='fixture')

    def collect_pool(self, count):
        assert count == 2560
        self.pools.append(count); self.consumed += count
        # The next pool deliberately revisits the same question IDs. Distinct
        # occurrence IDs must then come from candidate_offset, not new content.
        indices = torch.arange(count, dtype=torch.long)
        tensors = dict(input_ids=torch.stack((indices+100, indices*0+7, indices*0+9), dim=1),
            attention_mask=torch.ones(count, 3, dtype=torch.long),
            position_ids=torch.arange(3).expand(count, 3).clone())
        non_tensors = dict(index=np.array([dict(question=i) for i in range(count)], dtype=object),
            candidate_stable_id=np.array(['question-'+str(i) for i in range(count)], dtype=object),
            candidate_original_question=np.array(['Original question '+str(i) for i in range(count)], dtype=object),
            candidate_fixture_ordinal=np.array(range(count), dtype=object))
        return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors), list(range(count))

    def materialize(self, parents, ordinals):
        self.materializations.append(tuple(ordinals))
        return subset(parents, ordinals)

    def state(self):
        return dict(candidate_order=list(range(2560)), candidate_cursor=2560,
            epoch=len(self.pools)-1, consumed_source_rows=self.consumed,
            pending_source=None, pending_offset=0)

    def parents(self, *args):
        raise AssertionError('Fixed2560 collection fell back to early-stop legacy parents()')


class FixedPoolManager:
    def __init__(self):
        self.config = NS(max_start_length=2048)
        self.timing_raw = {}; self.calls = []; self.originals = {}; self.snapshots = []

    def run_llm_loop(self, *, gen_batch, initial_input_ids):
        assert len(gen_batch) == 2560
        assert torch.equal(initial_input_ids, gen_batch.batch['input_ids'])
        call = len(self.calls); self.calls.append(len(gen_batch))
        identity = copy.deepcopy(gen_batch.meta_info['behavior_identity'])
        self.snapshots.append(identity)
        width = 2 + call % 3
        row = torch.arange(len(gen_batch), dtype=torch.long)
        responses = row[:, None] * 11 + torch.arange(width)[None, :] + call * 100000
        prompts = gen_batch.batch['input_ids'].clone()
        probs = -(row[:, None].double()+1) / 10000 - torch.arange(width)[None, :].double()/1000 - call
        tensors = dict(prompts=prompts, responses=responses,
            responses_with_info_mask=responses.clone(), input_ids=torch.cat((prompts, responses), dim=1),
            sampling_log_probs=probs, sampling_mask=torch.ones_like(responses, dtype=torch.bool),
            attention_mask=torch.ones(len(row), 3+width, dtype=torch.long),
            info_mask=torch.cat((torch.zeros_like(prompts), torch.ones_like(responses)), dim=1),
            position_ids=torch.arange(3+width).expand(len(row), 3+width).clone())
        ids = gen_batch.non_tensor_batch['behavior_row_id'].tolist()
        for i, identity_id in enumerate(ids):
            assert identity_id not in self.originals, 'Candidate occurrence IDs collided across pools'
            self.originals[identity_id] = {name: value[i].clone() for name, value in tensors.items()}
        meta = {key: [call*2560+i for i in range(2560)] for key in ROW_META}
        meta['actual_retrieval_stats'] = [1]*2560
        meta['behavior_identity'] = identity
        self.timing_raw = {'generation_fixture': .125}
        return DataProto.from_dict(tensors=tensors,
            non_tensors=dict(behavior_trajectory_id=np.array(ids, dtype=object)), meta_info=meta)


class FixedPoolCollectionTests(unittest.TestCase):
    def run_collection(self, correctness, *, wrapper=False):
        stream = FixedPoolStream(); manager = FixedPoolManager(); captures = []
        trainer = NS(global_steps=1, tokenizer=NS(pad_token_id=0),
            config=NS(data=dict(selection_candidate_questions=2560,
                selection_selected_questions=512, selection_fixed_candidate_budget=True)),
            actor_rollout_wg=NS(capture_mixed_state=lambda path: captures.append(path) or []),
            reward_fn=NS(expected_probe_version='fixture:0'))
        attempt = dict(attempt_id='a'*32, attempt_index=1)
        run = dict(run_id='cpu-fixed2560', source_commit='b'*40, source_sha256={},
            model_revision='fixture-model', budget=dict(common_manifest_sha256='fixture-data'),
            policy_safety_mode='inline_first_five')
        def strict(parent, output, tokenizer):
            pool = (len(manager.calls)-1)//5
            return [correctness(pool, int(i)) for i in parent.non_tensor_batch['candidate_fixture_ordinal']]
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder); (base/'logs').mkdir()
            (base/'logs/run-manifest.json').write_text(json.dumps(run))
            with patch.dict(os.environ, SEARCH_G1_BASE=folder), \
                    patch('reproduction.common8.behavior_runtime._native_run', return_value=(base, run, 'cpu-only')), \
                    patch.object(runtime, 'strict_vectors', side_effect=strict), \
                    contextlib.redirect_stdout(io.StringIO()):
                # Both wrappers are real. In particular, neither MixedFirstSelector
                # nor _collect_once is replaced by a preselected fixture result.
                fn = runtime.collect if wrapper else runtime._collect_once
                parents, output, identity, audit = fn(trainer, manager, stream, attempt)
            paths = list((base/'artifacts/mixed-candidate-pools').glob('*/selection-failure.json'))
            failures = [json.loads(path.read_text()) for path in paths]
        return NS(parents=parents, output=output, identity=identity, audit=audit,
            manager=manager, stream=stream, captures=captures, failures=failures)

    def assert_original_selected_rows(self, result, ordinals):
        self.assertEqual(len(result.parents), 2560); self.assertEqual(len(result.output), 2560)
        self.assertEqual(result.parents.non_tensor_batch['candidate_fixture_ordinal'].tolist(),
                         [i for i in ordinals for _ in range(5)])
        ids = result.output.non_tensor_batch['behavior_trajectory_id'].tolist()
        self.assertEqual(ids, result.identity['trajectory_ids'])
        self.assertEqual(len(set(ids)), 2560)
        for row, identity_id in enumerate(ids):
            original = result.manager.originals[identity_id]
            for name in ('responses', 'responses_with_info_mask', 'sampling_log_probs', 'sampling_mask'):
                width = original[name].shape[0]
                self.assertTrue(torch.equal(result.output.batch[name][row, :width], original[name]), name)
                if result.output.batch[name].shape[1] > width:
                    self.assertFalse(bool(result.output.batch[name][row, width:].any()), name)
            width = original['input_ids'].shape[0]
            self.assertTrue(torch.equal(result.output.batch['input_ids'][row, :width], original['input_ids']))
        selected = [group['ordinal'] for group in result.audit['groups'] if group['selected']]
        self.assertEqual(selected, ordinals)

    def test_fixed2560_generates_all12800_even_after_first512_are_mixed(self):
        result = self.run_collection(lambda pool, ordinal: [1, 0, 1, 0, 1])
        self.assertEqual(result.manager.calls, [2560]*5)
        self.assertEqual(len(result.captures), 10)
        self.assertEqual(result.stream.pools, [2560])
        self.assertEqual(result.audit['attempted_candidates'], 2560)
        self.assertEqual(result.audit['raw_generated_trajectories'], 12800)
        self.assertEqual(result.audit['selected_mixed'], 512)
        self.assertEqual(result.audit['selected_all_correct_fallback'], 0)
        self.assertEqual(len({row['policy_snapshot_id'] for row in result.manager.snapshots}), 1)
        self.assertEqual([rows[0] for rows in result.stream.materializations], [0, 512, 1024, 1536, 2048])
        self.assert_original_selected_rows(result, list(range(512)))
        self.assertFalse(bool(result.parents.batch['selection_fallback'].any()))

    def test_ordinal_mixed_then_correct_fallback_reuses_original_rows_across_chunks(self):
        def correctness(pool, ordinal):
            return [1, 0, 0, 0, 0] if 100 <= ordinal < 500 else [1]*5 if 1000 <= ordinal < 1200 else [0]*5
        result = self.run_collection(correctness)
        selected = list(range(100, 500)) + list(range(1000, 1112))
        self.assert_original_selected_rows(result, selected)
        self.assertEqual(result.manager.calls, [2560]*5)
        self.assertEqual(result.audit['selected_mixed'], 400)
        self.assertEqual(result.audit['selected_all_correct_fallback'], 112)
        self.assertEqual(int(result.parents.batch['selection_fallback'].sum()), 112*5)
        self.assertEqual(result.audit['raw_generated_trajectories'], 12800)

    def test_insufficient_pool_retries_complete_new_pool_without_id_collisions(self):
        def correctness(pool, ordinal):
            if pool == 0: return [1, 0, 0, 0, 0] if ordinal < 100 else [0]*5
            return [1, 0, 1, 0, 0]
        result = self.run_collection(correctness, wrapper=True)
        self.assertEqual(result.manager.calls, [2560]*10)
        self.assertEqual(result.stream.pools, [2560, 2560])
        self.assertEqual(result.stream.consumed, 5120)
        self.assertEqual(result.audit['pool_retry'], 1)
        self.assertEqual(result.audit['raw_generated_trajectories'], 25600)
        self.assertEqual(result.audit['raw_actual_retrieval_calls'], 25600)
        self.assertAlmostEqual(result.audit['candidate_generation_timings']['generation_fixture'], 1.25)
        self.assertEqual(len(result.audit['discarded_insufficient_pools']), 1)
        self.assertEqual(len(result.failures), 1)
        self.assertFalse(result.failures[0]['optimizer_started'])
        self.assertEqual(result.failures[0]['raw_generated_trajectories'], 12800)
        first_ids = {value for snap in result.manager.snapshots[:5] for value in snap['trajectory_ids']}
        second_ids = {value for snap in result.manager.snapshots[5:] for value in snap['trajectory_ids']}
        self.assertEqual(len(first_ids), 12800); self.assertEqual(len(second_ids), 12800)
        self.assertFalse(first_ids & second_ids)
        # The questions are identical across retries; only occurrence offsets
        # distinguish the pool identities, proving offset2560 is really used.
        first_parent = result.manager.snapshots[0]['parent_group_ids'][0]
        second_parent = result.manager.snapshots[5]['parent_group_ids'][0]
        self.assertTrue(first_parent.startswith('0-'))
        self.assertTrue(second_parent.startswith('2560-'))
        self.assertEqual(first_parent.split('-', 1)[1], second_parent.split('-', 1)[1])
        self.assert_original_selected_rows(result, list(range(512)))

if __name__=='__main__':unittest.main()
