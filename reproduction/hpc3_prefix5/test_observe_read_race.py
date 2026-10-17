"""A transient ledger publication must not hide persistent missing failure evidence."""
import ast
import io
import json
import unittest

from .observe import QUERY


class ChangingPath:
    def __init__(self, contents, missing_opens=0):
        self.contents, self.missing_opens, self.opens = contents, missing_opens, 0

    def open(self, mode):
        self.opens += 1
        if self.opens <= self.missing_opens:
            raise FileNotFoundError('simulated missing publication')
        return io.BytesIO(self.contents)


class ObserveReadRaceTest(unittest.TestCase):
    def setUp(self):
        module = ast.parse(QUERY)
        function = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == 'read')
        self.sleeps = []
        class Clock:
            sleep = self.sleeps.append
        namespace = dict(json=json, time=Clock())
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<actual-observer-read>', 'exec'), namespace)
        self.read = namespace['read']

    def test_required_transient_missing_recovers(self):
        path = ChangingPath(b'{"jobs":{}}', missing_opens=1)
        self.assertEqual(self.read(path, required=True), dict(jobs={}))
        self.assertEqual(path.opens, 2)
        self.assertEqual(self.sleeps, [.05])

    def test_required_persistent_missing_is_visible_and_bounded(self):
        path = ChangingPath(b'{}', missing_opens=10)
        with self.assertRaises(FileNotFoundError):
            self.read(path, required=True)
        self.assertEqual(path.opens, 3)

    def test_optional_missing_is_not_a_training_failure(self):
        self.assertIsNone(self.read(ChangingPath(b'{}', missing_opens=1)))

    def test_corrupt_json_is_not_swallowed(self):
        with self.assertRaises(json.JSONDecodeError):
            self.read(ChangingPath(b'{'))

    def test_metadata_size_limit_is_retained(self):
        with self.assertRaises(ValueError):
            self.read(ChangingPath(b'{}'), limit=1)


if __name__ == '__main__':
    unittest.main()
