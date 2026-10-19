import ast
import unittest

from reproduction.grounding_eval.a800_retriever import single_gpu_entrypoint


class SingleGpuLaunchTests(unittest.TestCase):
    SOURCE = '''def main():
    parser.add_argument('--gpu-shards', type=int, choices=(2,8), default=8)
    if retriever.index.count() != args.gpu_shards or retriever.index.d != 768:
        raise ValueError('shards')
    retriever.batch_search(query_list=queries, num=3, return_score=True)
'''

    def test_only_launch_guards_change(self):
        adapted = single_gpu_entrypoint(self.SOURCE)
        tree = ast.parse(adapted)
        self.assertIn('choices=(1,2,8)', adapted)
        self.assertIn("hasattr(retriever.index, 'count')", adapted)
        self.assertEqual(ast.dump(ast.parse(self.SOURCE).body[0].body[-1]),
                         ast.dump(tree.body[0].body[-1]))

    def test_changed_upstream_fails_closed(self):
        with self.assertRaises(ValueError):
            single_gpu_entrypoint(self.SOURCE.replace('choices=(2,8)', 'choices=(2, 8)'))
        with self.assertRaises(ValueError):
            single_gpu_entrypoint(self.SOURCE + self.SOURCE)


if __name__ == '__main__':
    unittest.main()
