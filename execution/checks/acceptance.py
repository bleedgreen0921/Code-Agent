"""Fixed assertions run only in the supervisor, never in the candidate interpreter."""
import unittest

TEST_IDS = ('zero', 'normal', 'empty', 'over_length', 'negative', 'no_mutation')


def suite(invoke):
    class FirstN(unittest.TestCase):
        def id(self):
            return 'first_n.' + self._testMethodName.removeprefix('test_')

        def value(self, items, n):
            reply = invoke(items, n)
            if reply['error'] is not None:
                raise RuntimeError(reply['error'])
            self.assertEqual(reply['value_type'], 'list')
            return reply

        def test_zero(self):
            self.assertEqual(self.value([1, 2, 3], 0)['value'], [])

        def test_normal(self):
            self.assertEqual(self.value([1, 2, 3], 2)['value'], [1, 2])

        def test_empty(self):
            self.assertEqual(self.value([], 2)['value'], [])

        def test_over_length(self):
            self.assertEqual(self.value([1, 2], 5)['value'], [1, 2])

        def test_negative(self):
            self.assertEqual(invoke([1, 2, 3], -1)['error'], 'ValueError')

        def test_no_mutation(self):
            reply = self.value([3, 1, 2], 2)
            self.assertEqual(reply['items'], [3, 1, 2])
            self.assertEqual(reply['value'], [3, 1])

    return unittest.TestSuite(FirstN('test_' + name) for name in TEST_IDS)
