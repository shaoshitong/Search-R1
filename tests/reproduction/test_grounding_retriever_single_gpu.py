import ast
import unittest

from types import SimpleNamespace
from reproduction.grounding_eval.a800_retriever import single_gpu_entrypoint, bounded_single_gpu_clone


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

    def test_bounded_clone_keeps_flat_vector_order_and_fp16(self):
        class Vectors:
            def __init__(self, start, count):
                self.shape = (count, 768)
                self.values = list(range(start, start+count))
        class Flat:
            d = 768
            metric_type = 0
            ntotal = 10
            def reconstruct_n(self, start, count):
                return Vectors(start, count)
        class Resource:
            def setTempMemory(self, size):
                self.temp = size
        class Gpu:
            d = 768
            metric_type = 0
            ntotal = 0
            def __init__(self): self.values=[]
            def add(self, vectors):
                self.values.extend(vectors.values)
                self.ntotal += len(vectors.values)
        gpu = Gpu()
        options = SimpleNamespace(useFloat16=True)
        captured=[]
        def clone(resource, device, empty, co):
            captured.append((device, co.useFloat16, co.reserveVecs))
            return gpu
        api = SimpleNamespace(downcast_index=lambda x:x, get_num_gpus=lambda:1,
            IndexFlat=Flat, METRIC_INNER_PRODUCT=0, METRIC_L2=1,
            IndexFlatIP=lambda d:None, StandardGpuResources=Resource,
            GpuClonerOptions=SimpleNamespace, index_cpu_to_gpu=clone)
        progress=[]
        actual=bounded_single_gpu_clone(api, Flat(), options,
            lambda n,total:progress.append((n,total)), batch_vectors=3)
        self.assertIs(actual,gpu)
        self.assertEqual(gpu.values,list(range(10)))
        self.assertEqual(captured,[(0,True,10)])
        self.assertEqual(progress,[(3,10),(6,10),(9,10),(10,10)])
        options.useFloat16=False
        with self.assertRaises(ValueError):
            bounded_single_gpu_clone(api, Flat(), options, lambda *a:None)


if __name__ == '__main__':
    unittest.main()
