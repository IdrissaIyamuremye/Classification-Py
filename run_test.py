import sys
import os
import unittest

# Add project root to sys.path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# Import your specific test case
from test.Classifier.XGBoostTest import XGBoostTest

if __name__ == '__main__':
    # Load only XGBoostTest
    suite = unittest.TestLoader().loadTestsFromTestCase(XGBoostTest)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)