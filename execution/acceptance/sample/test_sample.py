"""Deployment-owned example acceptance. Never imported from the candidate tree."""
import unittest
from acceptance_support import invoke


class TestAnswer(unittest.TestCase):
    def test_value(self):
        self.assertEqual(invoke('answer'), 1)
