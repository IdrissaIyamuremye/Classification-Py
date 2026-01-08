"""
XGBoost Decision Tree
"""

import random
from typing import List
from Classification.Instance.Instance import Instance
from Classification.InstanceList.InstanceList import InstanceList
from Classification.Model.DecisionTree.DecisionTree import DecisionTree
from Classification.Model.DecisionTree.XGBoostNode import XGBoostNode
from Classification.Parameter.XGBoostParameter import XGBoostParameter


class XGBoostTree(DecisionTree):
    """
    Single tree in the XGBoost ensemble.
    """
    
    def __init__(self, data: InstanceList, 
                 gradients: List[float], 
                 hessians: List[float],
                 instance_indices: List[int],
                 parameter: XGBoostParameter):
        """
        Initialize XGBoost tree with gradient information.
        """
        # Determine feature subset for this tree (colsample_bytree)
        feature_subset = None
        if parameter and parameter.getColsampleByTree() < 1.0:
            n_features = data.get(0).attributeSize()
            n_sample = max(1, int(n_features * parameter.getColsampleByTree()))
            feature_subset = random.sample(range(n_features), n_sample)
        
        root = XGBoostNode(data, gradients, hessians, instance_indices, 
                          None, parameter, 0, feature_subset)
        self._DecisionTree__root = root
    
    def predictValue(self, instance: Instance) -> float:
        """
        Predict the raw value for gradient boosting.
        """
        return self._DecisionTree__root.predictLeafValue(instance)