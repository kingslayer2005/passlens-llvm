import unittest
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.targets import build_target_labels

class TestTargets(unittest.TestCase):
    def test_build_target_labels_non_loop(self):
        data = {
            'outcome': ['success', 'success', 'success', 'error'],
            'inst_before': [100, 100, 100, 100],
            'inst_after': [90, 105, 100, 0],
            'inst_control': [100, 100, 100, 100],
            'fired': [1, 1, 0, 0]
        }
        df = pd.DataFrame(data)
        
        # Test beneficial
        df_test = df.copy()
        labels = build_target_labels(df_test, 'instcombine', 'beneficial', 0.01)
        self.assertEqual(len(labels), 3) # error row removed
        self.assertEqual(list(labels), [True, False, False])
        
        # Test harmful
        df_test = df.copy()
        labels = build_target_labels(df_test, 'instcombine', 'harmful')
        self.assertEqual(list(labels), [False, True, False])
        
        # Test fired
        df_test = df.copy()
        labels = build_target_labels(df_test, 'instcombine', 'fired')
        self.assertEqual(list(labels), [True, True, False])

    def test_build_target_labels_loop(self):
        data = {
            'outcome': ['success', 'success', 'success'],
            'inst_before': [100, 100, 100],
            'inst_after': [105, 120, 110],
            'inst_control': [110, 110, 110], # loop-simplify adds 10
            'fired': [1, 1, 0]
        }
        df = pd.DataFrame(data)
        
        # Test beneficial (needs (110 - inst_after)/110 >= 0.01)
        # Row 0: (110 - 105) / 110 = 0.045 >= 0.01 -> True
        # Row 1: (110 - 120) / 110 < 0 -> False
        # Row 2: (110 - 110) / 110 = 0 -> False
        df_test = df.copy()
        labels = build_target_labels(df_test, 'licm', 'beneficial', 0.01)
        self.assertEqual(list(labels), [True, False, False])
        
        # Test harmful (inst_after > 110)
        # Row 0: 105 > 110 -> False
        # Row 1: 120 > 110 -> True
        # Row 2: 110 > 110 -> False
        df_test = df.copy()
        labels = build_target_labels(df_test, 'licm', 'harmful')
        self.assertEqual(list(labels), [False, True, False])

if __name__ == '__main__':
    unittest.main()
