import sys
sys.path.insert(0, '../')

from Classification.Attribute.AttributeType import AttributeType
from Classification.DataSet.DataDefinition import DataDefinition
from Classification.DataSet.DataSet import DataSet
from Classification.Model.XGBoost.XGBoost import XGBoostClassifier, XGBoostParameter

## Run all datasets
datasets = {
    'iris': (4 * [AttributeType.CONTINUOUS], "../datasets/iris.data"),
    'bupa': (6 * [AttributeType.CONTINUOUS], "../datasets/bupa.data"),
    'dermatology': (34 * [AttributeType.CONTINUOUS], "../datasets/dermatology.data"),
    'car': (6 * [AttributeType.DISCRETE], "../datasets/car.data"),
    'tictactoe': (9 * [AttributeType.DISCRETE], "../datasets/tictactoe.data"),
}

xgboostParameter = XGBoostParameter(seed=1, n_estimators=50, max_depth=4, learning_rate=0.3)

print("\nFinal test of all datasets:")
print("="*60)
for name, (attr_types, file_path) in datasets.items():
    dataDefinition = DataDefinition(attr_types)
    dataset = DataSet(dataDefinition, ",", file_path)
    data = dataset.getInstanceList()
    
    xgboost = XGBoostClassifier()
    xgboost.train(data, xgboostParameter)
    result = xgboost.test(data)
    error_pct = result.getErrorRate() * 100
    
    status = "✓ PASS" if error_pct < 0.01 else "✗ FAIL"
    print(f"{name:15s}: {error_pct:6.4f}% [{status}]")