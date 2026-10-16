import unittest

from .pool import initial_state, reserve_indices


class CandidatePoolTest(unittest.TestCase):
    def test_fresh_first_outer_does_not_inherit_teacher_cursor(self):
        before = initial_state(168510, 1729)
        indices, cycles, after = reserve_indices(before, 1)
        self.assertEqual(len(set(indices)), 5120)
        self.assertEqual(after['consumed_candidates'], 5120)
        self.assertEqual(after['next_outer'], 2)
        self.assertEqual(cycles, [0] * 5120)
        self.assertEqual(before['cursor'], 0)

    def test_no_cross_outer_repeat_before_cycle_exhaustion(self):
        state = initial_state(168510)
        first, _, state = reserve_indices(state, 1)
        second, _, _ = reserve_indices(state, 2)
        self.assertFalse(set(first) & set(second))

    def test_boundary_outer_contains_no_duplicate_and_no_omission(self):
        state = initial_state(7)
        a, _, state = reserve_indices(state, 1, 5)
        b, cycles, state = reserve_indices(state, 2, 5)
        self.assertEqual(len(set(b)), 5)
        self.assertEqual(len(set(a + b[:2])), 7)
        self.assertEqual(cycles, [0, 0, 1, 1, 1])
        self.assertEqual(state['consumed_candidates'], 10)

    def test_resume_reservation_reproduces_question_order(self):
        state = initial_state(13)
        _, _, state = reserve_indices(state, 1, 5)
        self.assertEqual(reserve_indices(state, 2, 5), reserve_indices(dict(state), 2, 5))
        with self.assertRaises(ValueError):
            reserve_indices(state, 3, 5)


if __name__ == '__main__':
    unittest.main()
