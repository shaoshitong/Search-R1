import copy
import unittest

from .checkpoint_cpu_once import validate_probe_clock, validate_selection
from .diversity import select_groups, summarize
from .test_diversity import fixture


class ActualMilestoneContractTests(unittest.TestCase):
    def test_diversity_recomputed_from_all1024_original_labels(self):
        selection, labels = fixture([0]*256+[1]*256+[2]*256+[3]*256)
        for tid, row in labels.items(): row['trajectory_id'] = tid
        chosen, all_groups = select_groups(selection, labels)
        hashes = {key: {'sha256': key} for key in ('selection', 'rollout', 'actor_selection')}
        selected = dict(selected=chosen, continued1024=summarize(all_groups), actor512=summarize(chosen),
            prefix_selection_sha256='selection', full_rollout_sha256='rollout',
            outcome_selection='population_strict_EM_variance_K_times_5minusK_over25',
            tie_break='frozen_prefix_q_descending_then_original_candidate_ordinal',
            probe_uses_all_continued1024=True, uncontinued_candidate_questions=4096,
            all_candidate_accuracy_available=False)
        actor_rollout = dict(actor_selection_metrics=selected, actor_selection_sha256='actor_selection',
            source_full_rollout_sha256='rollout', questions=512, trajectories=2560,
            actor_version='actor4', raw_native_records_preserved=True)
        args = (selection, list(labels.values()), selected, {'actor_version': 'actor4'},
                actor_rollout, {'actor_selection': selected}, hashes)
        self.assertTrue(validate_selection(*args)['deterministic_population_variance_top512_recomputed'])
        selected['selected'] = list(reversed(chosen))
        with self.assertRaises(ValueError): validate_selection(*args)

    def test_twenty_synchronous_steps_true_mean_and_prediction_clock(self):
        ranks = [[] for _ in range(8)]; globals_ = []; extra = 0
        for step in range(1, 21):
            n = 52 if step%5 == 0 else 51
            counts = [6]*8
            for j in range(n-48): counts[(extra+j)%8] += 1
            extra = (extra+n-48)%8
            for rank, count in enumerate(counts):
                ranks[rank].append(dict(rank=rank, outer=5, step=step, actual_adam_steps=80+step,
                    actual_learning_rate=1e-6, local_questions=count, local_trajectories=count*5,
                    global_questions=n, trajectory_bce=.75, group_bce=.625, total_loss=1.375,
                    backward_loss=1.375*8*count/n))
            globals_.append(dict(outer=5, step=step, actual_adam_steps_per_rank=80+step,
                actual_learning_rate=1e-6, global_questions=n, global_trajectories=n*5,
                loss_prediction_clock='before_this_step_adam_update', trajectory_bce=.75,
                group_bce=.625, total_loss=1.375))
        metrics = dict(global_step_loss_curve={'points': globals_},
            diagnostic_semantics='same_post_update_head_complete_selected1024G5_and_fixed512G5',
            train_online_trajectory_bce=.75, train_online_group_bce=.625, train_online_total_loss=1.375)
        self.assertEqual(validate_probe_clock(metrics, ranks, globals_, 5)['trajectory_denominator'], 5120)
        wrong = copy.deepcopy(ranks); wrong[0][-1]['actual_adam_steps'] = 99
        with self.assertRaises(ValueError): validate_probe_clock(metrics, wrong, globals_, 5)
        wrong = copy.deepcopy(metrics); wrong['train_online_total_loss'] *= 2
        with self.assertRaises(ValueError): validate_probe_clock(wrong, ranks, globals_, 5)


if __name__ == '__main__': unittest.main()
