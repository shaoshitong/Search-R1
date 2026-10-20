"""Independent double-precision finite-combination value/gradient checks."""
import itertools,unittest
import torch
import torch.nn.functional as F
from .exact_uniform_g5_loss import exact_uniform_g5_loss,mixed_combinatorial_mass,choose


def groups(K,size=10):
    return [dict(qid='q%d'%q,trajectory_ids=['q%d:t%d'%(q,i) for i in range(size)],labels=[int(i<k) for i in range(size)]) for q,k in enumerate(K)]


def brute(logits,source,a=1.,b=1.):
    """No production event helper, validator, analytic BCE or custom VJP."""
    totals=[]
    for z,g in zip(logits,source):
        own=[]
        for ix in itertools.combinations(range(z.numel()),5):
            zz=z[list(ix)];yy=torch.tensor([g['labels'][i] for i in ix],dtype=z.dtype);k=int(yy.sum());lp=F.logsigmoid(zz);ln=F.logsigmoid(-zz)
            if k==0:logevent=ln.sum()
            elif k==5:logevent=lp.sum()
            else:
                patterns=[]
                for bit in range(1,31):patterns.append(sum(lp[j] if bit&(1<<j) else ln[j] for j in range(5)))
                logevent=torch.logsumexp(torch.stack(patterns),0)
            own.append(a*F.binary_cross_entropy_with_logits(zz,yy)-b*logevent)
        totals.append(torch.stack(own).mean())
    return torch.stack(totals).mean()


class ExactUniformTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);torch.manual_seed(1729)
    def test_double_value_gradient_independent_G10_bruteforce(self):
        gs=groups([0,1,5,9,10]);z=torch.randn(5,10,dtype=torch.float64,requires_grad=True)*2;got=exact_uniform_g5_loss(z,gs,trajectory_weight=.7,three_event_weight=1.3,combination_chunk=19);gg=torch.autograd.grad(got['loss'],z)[0];want=brute(z,gs,.7,1.3);wg=torch.autograd.grad(want,z)[0];torch.testing.assert_close(got['loss'],want,rtol=2e-13,atol=2e-13);torch.testing.assert_close(gg,wg,rtol=2e-12,atol=2e-13)
    def test_exact_combinatorial_mixed_mass(self):
        for size in (10,20):
            for k in range(size+1):
                y=[1]*k+[0]*(size-k);m=mixed_combinatorial_mass(y);count=sum(0<sum(y[i] for i in ix)<5 for ix in itertools.combinations(range(size),5));self.assertEqual(m['Mixed'],count);self.assertEqual(m['total'],choose(size,5));self.assertEqual(m['allcorrect']+m['allwrong']+count,m['total'])
    def test_permutation_invariance_values_gradients_and_chunk_size(self):
        gs=groups([2,7]);z=torch.randn(2,10,dtype=torch.float64,requires_grad=True);a=exact_uniform_g5_loss(z,gs,combination_chunk=1);ga=torch.autograd.grad(a['loss'],z)[0];perm=torch.randperm(10);pg=[dict(g,trajectory_ids=[g['trajectory_ids'][i] for i in perm],labels=[g['labels'][i] for i in perm]) for g in gs];p=z.detach()[:,perm].requires_grad_(True);b=exact_uniform_g5_loss(p,pg,combination_chunk=127);gb=torch.autograd.grad(b['loss'],p)[0];torch.testing.assert_close(a['loss'],b['loss'],rtol=2e-13,atol=2e-13);torch.testing.assert_close(ga[:,perm],gb,rtol=2e-12,atol=2e-13)
    def test_G20_finite_gradient_and_bounded_combinatorial_memory_contract(self):
        gs=groups([0,1,10,19,20],20);z=torch.linspace(-1000,1000,100,dtype=torch.float64).reshape(5,20).requires_grad_(True);r=exact_uniform_g5_loss(z,gs,combination_chunk=31);grad=torch.autograd.grad(r['loss'],z)[0];self.assertTrue(torch.isfinite(r['loss']) and torch.isfinite(grad).all());self.assertEqual(r['combinations_per_question'],15504);self.assertLessEqual(r['maximum_resident_G5_rows'],5*31);self.assertEqual(r['retained_loss_gradient_elements'],100);self.assertEqual(r['inter_view_pairs_allocated'],0);self.assertEqual(r['per_view_python_identity_metadata_allocated'],0)
    def test_uniform_source_BCE_analytic_identity(self):
        gs=groups([1,5]);z=torch.randn(2,10,dtype=torch.float64,requires_grad=True);r=exact_uniform_g5_loss(z,gs,three_event_weight=0);y=torch.tensor([g['labels'] for g in gs],dtype=z.dtype);want=F.binary_cross_entropy_with_logits(z,y);torch.testing.assert_close(r['loss'],want,rtol=1e-14,atol=1e-14);torch.testing.assert_close(torch.autograd.grad(r['loss'],z)[0],(z.sigmoid()-y)/20,rtol=1e-14,atol=1e-14)
    def test_unsupported_loss_and_fake_source_rejected(self):
        gs=groups([3]);z=torch.zeros(1,10)
        for key in ('binary_mixed_weight','within_rank_weight','inter_rank_weight','stat_mean_weight','stat_variance_weight'):
            with self.assertRaises(ValueError):exact_uniform_g5_loss(z,gs,**{key:.1})
        bad=groups([3]);bad[0]['trajectory_ids'][1]=bad[0]['trajectory_ids'][0]
        with self.assertRaises(ValueError):exact_uniform_g5_loss(z,bad)
        with self.assertRaises(ValueError):exact_uniform_g5_loss(z,gs,combination_chunk=257)

if __name__=='__main__':unittest.main()
