import unittest
import numpy as np
from Classification.Model.XGBoost.XGBoostRegression import MyXGBoostRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


class TestXGBoostRegression(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load dataset
        data = fetch_california_housing()
        X, y = data.data, data.target

        # Train / test split
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train model (small & fast for unit tests)
        cls.model = MyXGBoostRegressor(
            n_estimators=10,
            max_depth=3,
            learning_rate=0.1
        )
        cls.model.fit(cls.X_train, cls.y_train)

    def test_model_created(self):
        self.assertIsNotNone(self.model)

    def test_predict_runs(self):
        y_pred = self.model.predict(self.X_test)
        self.assertEqual(len(y_pred), len(self.y_test))

    def test_predictions_are_finite(self):
        y_pred = self.model.predict(self.X_test)
        self.assertTrue(np.isfinite(y_pred).all())

    def test_predictions_are_numeric(self):
        y_pred = self.model.predict(self.X_test)
        self.assertTrue(np.issubdtype(y_pred.dtype, np.number))


if __name__ == "__main__":
    unittest.main()
