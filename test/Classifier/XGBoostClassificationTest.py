import unittest
import numpy as np
from Classification.Model.XGBoost.XGBoostClassification import MyXGBoostClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


class TestXGBoostClassification(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load data
        data = load_breast_cancer()
        X, y = data.data, data.target

        # Train / test split
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train model (small + fast for unit tests)
        cls.clf = MyXGBoostClassifier(
            n_estimator=10,
            max_depth=3,
            learning_rate=0.1
        )
        cls.clf.fit(cls.X_train, cls.y_train)

    def test_model_created(self):
        self.assertIsNotNone(self.clf)

    def test_predict_runs(self):
        y_pred = self.clf.predict(self.X_test)
        self.assertEqual(len(y_pred), len(self.y_test))

    def test_predict_proba_runs(self):
       
        y_prob = self.clf.predict_proba(self.X_test)

        # Must return one probability per sample
        self.assertEqual(len(y_prob), len(self.y_test))

        # Probabilities must be between 0 and 1
        self.assertTrue((y_prob >= 0).all())
        self.assertTrue((y_prob <= 1).all())    


    def test_prediction_values_valid(self):
        y_pred = self.clf.predict(self.X_test)
        self.assertTrue(np.all(np.isin(y_pred, [0, 1])))


if __name__ == "__main__":
    unittest.main()
