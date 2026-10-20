import unittest
from . import peer_newseed_prefix_diagnostic as p


def row(j):
 tokens=list(range(20));tokens[9]=100+j
 return dict(tokens=tokens,position_ids=list(range(2,22)),segments=[dict(turn_index=0,policy_start=0,policy_length=20,position_start=2,request_length=2,context_tokens=[7,8]+tokens)],
  prompt=[7,8],label=j%2,response_index=j,question_hash='a'*64)


class PrefixDiagnosticTests(unittest.TestCase):
 def test_same_denominators_as_earlier_root_diagnostic(self):
  total=p.empty();p.add({'q':[row(j) for j in range(5)]},total);v=p.summarize(total)
  self.assertEqual(v['K_histogram'],[0,0,1,0,0,0]);a=v['statistics']['5'];b=v['statistics']['10']
  self.assertEqual(a['any_duplicate_prefix_questions'],1);self.assertEqual(a['equal_prefix_unordered_pairs'],10)
  self.assertEqual(a['all_within_question_unordered_pairs'],10);self.assertEqual(a['equal_prefix_opposite_EM_pairs'],6)
  self.assertEqual(a['all_within_question_correct_wrong_pairs'],6);self.assertEqual(a['true_Mixed_conflicting_prefix_question_fraction'],1.)
  self.assertEqual(a['token_equal_clusters_with_different_recorded_context_or_positions'],0)
  self.assertEqual(b['any_duplicate_prefix_questions'],0);self.assertEqual(b['equal_prefix_pair_fraction'],0.)
  self.assertAlmostEqual(a['token_exact_empirical_trajectory_BCE_lower_bound_nats'],p.entropy(2,5))
 def test_unequal_context_caveat_and_unpadded_short_tuple(self):
  group=[row(j) for j in range(5)];group[0]['position_ids'][0]+=1
  total=p.empty();p.add({'q':group},total);v=p.summarize(total)
  self.assertEqual(v['statistics']['5']['token_equal_clusters_with_different_recorded_context_or_positions'],1)
  short=row(0);short['tokens']=[1,2]
  total=p.empty();p.add({'q':[short]+[row(j) for j in range(1,5)]},total)
  self.assertEqual(p.summarize(total)['statistics']['5']['shorter_naturally_terminal_trajectories'],1)

if __name__=='__main__':unittest.main()
