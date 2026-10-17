import unittest

from sample import first_n


class FirstNTests(unittest.TestCase):
    def test_zero(self):
        self.assertEqual(first_n([1, 2, 3], 0), [])

    def test_normal(self):
        self.assertEqual(first_n([1, 2, 3], 2), [1, 2])

    def test_empty(self):
        self.assertEqual(first_n([], 2), [])

    def test_over_length(self):
        self.assertEqual(first_n([1, 2], 5), [1, 2])
