"""Recovery/identity invariants, without importing the historical GPU runtime."""
import json
import tempfile
import unittest
from pathlib import Path
from reproduction.grounding_eval.normal_hpc3 import atomic, digest, publish_normal, question_from_prompt, sha, execution_batches

class NormalSourceAcceptance(unittest.TestCase):
    def test_occurrences_and_actual_shard_hash_survive_publication(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); batches = root / 'normal-batches'; batches.mkdir()
            binding = {'checkpoint_hash': 'immutable'}
            # Repeated qid remains two different original occurrences.
            records = [{'qid': 'same', 'evaluation_ordinal': i, 'model_generated_tokens': 3,
                        'inference_generated_tokens': 5} for i in [7, 9]]
            source = batches / 'batch-000000.json'
            atomic(source, {'binding_hash': digest(binding), 'records': records,
                            'records_sha256': digest(records)})
            publish_normal(root, batches, binding, [7, 9])
            first = (root / 'normal.jsonl').read_bytes()
            publish_normal(root, batches, binding, [7, 9])
            self.assertEqual(first, (root / 'normal.jsonl').read_bytes())
            rows = [json.loads(l) for l in first.splitlines()]
            self.assertEqual([r['evaluation_ordinal'] for r in rows], [7, 9])
            self.assertTrue(all(r['source_shard_sha256'] == sha(source) for r in rows))
            self.assertEqual(json.loads((root/'normal-complete.json').read_text())['token_totals']['inference_generated_tokens'], 10)

    def test_foreign_configuration_cannot_complete(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); batches=root/'normal-batches';batches.mkdir()
            records=[{'evaluation_ordinal': 0}]
            atomic(batches/'batch-000000.json', {'binding_hash':'foreign', 'records':records,
                    'records_sha256':digest(records)})
            with self.assertRaises(ValueError):
                publish_normal(root,batches,{'checkpoint_hash':'ours'},[0])
            self.assertFalse((root/'normal-complete.json').exists())
            self.assertFalse((root/'normal.jsonl').exists())

    def test_adaptive_execution_bounds_do_not_change_job(self):
        job=dict(batch_size=64,max_batch_size=256,max_num_seqs=256)
        saved=dict(job)
        self.assertEqual(execution_batches(job,80*1024**3),(64,256))
        self.assertEqual(execution_batches(job,40*1024**3),(64,128))
        self.assertEqual(execution_batches(job,80*1024**3,16),(16,256))
        self.assertEqual(job,saved)
        with self.assertRaises(ValueError):execution_batches(job,80*1024**3,512)

    def test_question_is_not_instruction_example(self):
        self.assertEqual(question_from_prompt([{'role':'user','content':'Example: <answer> Beijing </answer>.\nQuestion: where is A?'}]), 'where is A?')
        with self.assertRaises(ValueError):
            question_from_prompt([{'role':'user','content':'No explicit question marker'}])

if __name__ == '__main__':
    unittest.main()
