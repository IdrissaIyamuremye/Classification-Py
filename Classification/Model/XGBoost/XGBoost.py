"""
XGBoost Classification Implementation 
This module provides an enhanced XGBoost gradient boosting classifier with bug fixes,
performance optimizations, and additional features.
"""

from math import log, exp
import random
from typing import List, Dict, Optional, Tuple
from Classification.Instance.Instance import Instance
from Classification.InstanceList.InstanceList import InstanceList
from Classification.Model.DecisionTree.DecisionTree import DecisionTree
from Classification.Model.DecisionTree.DecisionNode import DecisionNode
from Classification.Model.ValidatedModel import ValidatedModel
from Classification.Parameter.Parameter import Parameter


class XGBoostParameter(Parameter):
    """
    Parameter class for XGBoost algorithm.
    
    Attributes:
        __learning_rate: Step size shrinkage to prevent overfitting (0 < eta <= 1)
        __n_estimators: Number of boosting rounds (trees)
        __max_depth: Maximum depth of trees
        __min_child_weight: Minimum sum of instance weight needed in a child
        __gamma: Minimum loss reduction required for split
        __subsample: Subsample ratio of training instances (0 < ratio <= 1)
        __colsample_bytree: Subsample ratio of columns when constructing each tree
        __reg_lambda: L2 regularization term on weights
        __reg_alpha: L1 regularization term on weights
        __early_stopping_rounds: Stop if no improvement for N rounds
    """
    
    def __init__(self, seed: int, 
                 learning_rate: float = 0.3,
                 n_estimators: int = 100,
                 max_depth: int = 6,
                 min_child_weight: float = 0.0,
                 gamma: float = 0.0,
                 subsample: float = 1.0,
                 colsample_bytree: float = 1.0,
                 reg_lambda: float = 0.0,
                 reg_alpha: float = 0.0,
                 early_stopping_rounds: int = 10):
        """
        Initialize XGBoost parameters with validation.
        
        Raises:
            ValueError: If parameters are out of valid ranges
        """
        super().__init__(seed)
        
        # Validate parameters
        if not 0 < learning_rate <= 1:
            raise ValueError("learning_rate must be in (0, 1]")
        if n_estimators < 1:
            raise ValueError("n_estimators must be at least 1")
        if max_depth < 1:
            raise ValueError("max_depth must be at least 1")
        if min_child_weight < 0:
            raise ValueError("min_child_weight must be non-negative")
        if gamma < 0:
            raise ValueError("gamma must be non-negative")
        if not 0 < subsample <= 1:
            raise ValueError("subsample must be in (0, 1]")
        if not 0 < colsample_bytree <= 1:
            raise ValueError("colsample_bytree must be in (0, 1]")
        if reg_lambda < 0:
            raise ValueError("reg_lambda must be non-negative")
        if reg_alpha < 0:
            raise ValueError("reg_alpha must be non-negative")
        
        self.__learning_rate = learning_rate
        self.__n_estimators = n_estimators
        self.__max_depth = max_depth
        self.__min_child_weight = min_child_weight
        self.__gamma = gamma
        self.__subsample = subsample
        self.__colsample_bytree = colsample_bytree
        self.__reg_lambda = reg_lambda
        self.__reg_alpha = reg_alpha
        self.__early_stopping_rounds = early_stopping_rounds
    
    def getLearningRate(self) -> float:
        return self.__learning_rate
    
    def getNEstimators(self) -> int:
        return self.__n_estimators
    
    def getMaxDepth(self) -> int:
        return self.__max_depth
    
    def getMinChildWeight(self) -> float:
        return self.__min_child_weight
    
    def getGamma(self) -> float:
        return self.__gamma
    
    def getSubsample(self) -> float:
        return self.__subsample
    
    def getColsampleByTree(self) -> float:
        return self.__colsample_bytree
    
    def getRegLambda(self) -> float:
        return self.__reg_lambda
    
    def getRegAlpha(self) -> float:
        return self.__reg_alpha
    
    def getEarlyStoppingRounds(self) -> int:
        return self.__early_stopping_rounds


class XGBoostNode(DecisionNode):
    """
    Extended DecisionNode for XGBoost that supports gradient-based splits.
    Implements XGBoost-specific splitting logic with proper regularization.
    """
    
    EPSILON = 1e-10  # For numerical stability
    
    def __init__(self, data: InstanceList, 
                 gradients: List[float],
                 hessians: List[float],
                 instance_indices: List[int],
                 condition=None,
                 parameter: XGBoostParameter = None,
                 depth: int = 0,
                 feature_subset: Optional[List[int]] = None):
        """
        Initialize XGBoost decision node with gradient and hessian information.
        """
        self.__gradients = gradients
        self.__hessians = hessians
        self.__instance_indices = instance_indices
        self.__parameter = parameter
        self.__depth = depth
        self.__leaf_value = 0.0
        self.__feature_subset = feature_subset
        
        # Calculate leaf weight using gradient boosting formula
        sum_gradients = sum(gradients[i] for i in instance_indices)
        sum_hessians = sum(hessians[i] for i in instance_indices)
        
        if parameter is not None:
            reg_lambda = parameter.getRegLambda()
            reg_alpha = parameter.getRegAlpha()
        else:
            reg_lambda = 1.0
            reg_alpha = 0.0
        
        # Leaf weight formula with L1 regularization: -G / (H + lambda)
        if sum_hessians + reg_lambda > self.EPSILON:
            if reg_alpha > self.EPSILON:
                # Apply L1 regularization (soft thresholding)
                if sum_gradients > reg_alpha:
                    self.__leaf_value = -(sum_gradients - reg_alpha) / (sum_hessians + reg_lambda)
                elif sum_gradients < -reg_alpha:
                    self.__leaf_value = -(sum_gradients + reg_alpha) / (sum_hessians + reg_lambda)
                else:
                    self.__leaf_value = 0.0
            else:
                # No regularization - use pure gradient boosting formula
                denominator = sum_hessians + reg_lambda if reg_lambda > 0 else sum_hessians
                self.__leaf_value = -sum_gradients / max(denominator, 1e-10)
        
        # Initialize parent class attributes
        self.leaf = True
        self.children = []
        self._DecisionNode__condition = condition
        self._DecisionNode__class_label = None
        
        # Store class distribution for compatibility
        from Math.DiscreteDistribution import DiscreteDistribution
        self._DecisionNode__classLabelsDistribution = DiscreteDistribution()
        
        # Get labels only for instances in this node
        node_data = InstanceList()
        for idx in instance_indices:
            node_data.add(data.get(idx))
        
        labels = node_data.getClassLabels()
        for label in labels:
            self._DecisionNode__classLabelsDistribution.addItem(label)
        self._DecisionNode__class_label = InstanceList.getMaximum(labels) if labels else None
        
        # Check stopping criteria
        if depth >= (parameter.getMaxDepth() if parameter else 6):
            return
        
        if len(node_data.getDistinctClassLabels()) == 1:
            return
        
        # Only check min_child_weight if it's positive
        min_weight = parameter.getMinChildWeight() if parameter else 0.0
        if min_weight > 0 and sum_hessians < min_weight:
            return
        
        if len(instance_indices) < 2:
            return
        
        # Find best split using XGBoost gain calculation
        self.__findBestSplit(data, gradients, hessians, instance_indices, parameter)
    
    def __findBestSplit(self, data: InstanceList, 
                       gradients: List[float], 
                       hessians: List[float],
                       instance_indices: List[int],
                       parameter: XGBoostParameter) -> None:
        """
        Find the best split using XGBoost's gain formula.
        """
        from Classification.Attribute.ContinuousAttribute import ContinuousAttribute
        from Classification.Attribute.DiscreteAttribute import DiscreteAttribute
        from Classification.Model.DecisionTree.DecisionCondition import DecisionCondition
        
        best_gain = 0.0
        best_attribute = -1
        best_split_value = 0.0
        best_split_type = None
        best_left_indices = []
        best_right_indices = []
        
        reg_lambda = parameter.getRegLambda() if parameter else 1.0
        gamma = parameter.getGamma() if parameter else 0.0
        min_child_weight = parameter.getMinChildWeight() if parameter else 1.0
        
        sum_gradients = sum(gradients[i] for i in instance_indices)
        sum_hessians = sum(hessians[i] for i in instance_indices)
        
        # Determine which features to consider
        if self.__feature_subset is not None:
            features_to_try = self.__feature_subset
        else:
            features_to_try = range(data.get(0).attributeSize())
        
        # Try each attribute
        for attr_idx in features_to_try:
            attribute = data.get(instance_indices[0]).getAttribute(attr_idx)
            
            if isinstance(attribute, ContinuousAttribute):
                # Sort instances by attribute value
                sorted_indices = sorted(instance_indices, 
                                      key=lambda i: data.get(i).getAttribute(attr_idx).getValue())
                
                left_gradient = 0.0
                left_hessian = 0.0
                
                for i in range(len(sorted_indices) - 1):
                    idx = sorted_indices[i]
                    left_gradient += gradients[idx]
                    left_hessian += hessians[idx]
                    
                    current_value = data.get(idx).getAttribute(attr_idx).getValue()
                    next_value = data.get(sorted_indices[i + 1]).getAttribute(attr_idx).getValue()
                    
                    # Skip if values are identical
                    if abs(current_value - next_value) < self.EPSILON:
                        continue
                    
                    right_gradient = sum_gradients - left_gradient
                    right_hessian = sum_hessians - left_hessian
                    
                    # Check minimum child weight constraint (only if positive)
                    if min_child_weight > 0:
                        if left_hessian < min_child_weight or right_hessian < min_child_weight:
                            continue
                    
                    # Calculate gain with numerical stability
                    reg_term = reg_lambda if reg_lambda > 0 else 0
                    gain_left = (left_gradient * left_gradient) / (left_hessian + reg_term + self.EPSILON)
                    gain_right = (right_gradient * right_gradient) / (right_hessian + reg_term + self.EPSILON)
                    gain_parent = (sum_gradients * sum_gradients) / (sum_hessians + reg_term + self.EPSILON)
                    
                    gain = 0.5 * (gain_left + gain_right - gain_parent) - gamma
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_attribute = attr_idx
                        best_split_value = (current_value + next_value) / 2.0
                        best_split_type = 'continuous'
                        best_left_indices = sorted_indices[:i+1]
                        best_right_indices = sorted_indices[i+1:]
            
            elif isinstance(attribute, DiscreteAttribute):
                # Get unique values for this attribute among instances in this node
                value_set = set()
                for idx in instance_indices:
                    value_set.add(data.get(idx).getAttribute(attr_idx).getValue())
                
                value_list = list(value_set)
                
                # Try splitting on each value
                for value in value_list:
                    left_gradient = 0.0
                    left_hessian = 0.0
                    left_indices = []
                    right_indices = []
                    
                    for idx in instance_indices:
                        if data.get(idx).getAttribute(attr_idx).getValue() == value:
                            left_gradient += gradients[idx]
                            left_hessian += hessians[idx]
                            left_indices.append(idx)
                        else:
                            right_indices.append(idx)
                    
                    right_gradient = sum_gradients - left_gradient
                    right_hessian = sum_hessians - left_hessian
                    
                    # Check minimum child weight constraint (only if positive)
                    if min_child_weight > 0:
                        if left_hessian < min_child_weight or right_hessian < min_child_weight:
                            continue
                    
                    # Calculate gain
                    reg_term = reg_lambda if reg_lambda > 0 else 0
                    gain_left = (left_gradient * left_gradient) / (left_hessian + reg_term + self.EPSILON)
                    gain_right = (right_gradient * right_gradient) / (right_hessian + reg_term + self.EPSILON)
                    gain_parent = (sum_gradients * sum_gradients) / (sum_hessians + reg_term + self.EPSILON)
                    
                    gain = 0.5 * (gain_left + gain_right - gain_parent) - gamma
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_attribute = attr_idx
                        best_split_value = value
                        best_split_type = 'discrete'
                        best_left_indices = left_indices
                        best_right_indices = right_indices
        
        # Create children if a beneficial split was found
        if best_attribute != -1 and best_gain > self.EPSILON:
            self.leaf = False
            
            if best_split_type == 'continuous':
                self.children.append(
                    XGBoostNode(data, gradients, hessians, best_left_indices,
                              DecisionCondition(best_attribute, ContinuousAttribute(best_split_value), "<"),
                              parameter, self.__depth + 1, self.__feature_subset))
                self.children.append(
                    XGBoostNode(data, gradients, hessians, best_right_indices,
                              DecisionCondition(best_attribute, ContinuousAttribute(best_split_value), ">="),
                              parameter, self.__depth + 1, self.__feature_subset))
            
            elif best_split_type == 'discrete':
                # For discrete attributes, first child gets instances with the value
                # Second child gets instances without the value
                # Note: We store both sets of indices since DecisionCondition != may not work correctly
                left_child = XGBoostNode(data, gradients, hessians, best_left_indices,
                          DecisionCondition(best_attribute, DiscreteAttribute(best_split_value)),
                          parameter, self.__depth + 1, self.__feature_subset)
                left_child._XGBoostNode__is_discrete_left = True
                left_child._XGBoostNode__discrete_indices = set(best_left_indices)
                
                right_child = XGBoostNode(data, gradients, hessians, best_right_indices,
                          DecisionCondition(best_attribute, DiscreteAttribute(best_split_value)),
                          parameter, self.__depth + 1, self.__feature_subset)
                right_child._XGBoostNode__is_discrete_left = False
                right_child._XGBoostNode__discrete_indices = set(best_right_indices)
                
                self.children.append(left_child)
                self.children.append(right_child)
    
    def predictLeafValue(self, instance: Instance) -> float:
        """
        Predict the leaf value (weight) for the given instance.
        """
        if self.leaf:
            return self.__leaf_value
        else:
            # For discrete splits, we need special handling since != doesn't work
            # Check if any child has discrete split markers
            has_discrete = any(hasattr(child, '_XGBoostNode__is_discrete_left') for child in self.children)
            
            if has_discrete and len(self.children) == 2:
                # This is a discrete split - check which child matches
                left_child = self.children[0]
                right_child = self.children[1]
                
                # Get the attribute index from condition
                if left_child._DecisionNode__condition:
                    attr_idx = left_child._DecisionNode__condition._DecisionCondition__attribute_index
                    split_value = left_child._DecisionNode__condition._DecisionCondition__value.getValue()
                    instance_value = instance.getAttribute(attr_idx).getValue()
                    
                    # If instance value matches split value, go left; otherwise go right
                    if instance_value == split_value:
                        return left_child.predictLeafValue(instance)
                    else:
                        return right_child.predictLeafValue(instance)
            
            # Normal continuous split or fallback
            for node in self.children:
                if node._DecisionNode__condition.satisfy(instance):
                    return node.predictLeafValue(instance)
            
            # Fallback: if no condition matched, use first child
            if len(self.children) > 0:
                return self.children[0].predictLeafValue(instance)
            
            # Ultimate fallback
            return self.__leaf_value


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


class XGBoostClassifier(ValidatedModel):
    """
    XGBoost Gradient Boosting Classifier.
    """
    
    def __init__(self):
        """
        Initialize XGBoost classifier.
        """
        self.__trees = []
        self.__class_labels = []
        self.__n_classes = 0
        self.__base_score = 0.0
        self.__parameter = None
        self.__feature_importance = {}
        self.__training_history = []
    
    def __sigmoid(self, x: float) -> float:
        """
        Apply sigmoid function with numerical stability.
        """
        if x > 20:
            return 1.0
        if x < -20:
            return 0.0
        return 1.0 / (1.0 + exp(-x))
    
    def __softmax(self, scores: List[float]) -> List[float]:
        """
        Apply softmax function with numerical stability.
        """
        max_score = max(scores)
        exp_scores = [exp(s - max_score) for s in scores]
        sum_exp = sum(exp_scores)
        return [e / sum_exp for e in exp_scores]
    
    def train(self, trainSet: InstanceList, parameters: XGBoostParameter,
              validationSet: Optional[InstanceList] = None) -> None:
        """
        Train the XGBoost classifier.
        """
        self.__parameter = parameters
        self.__class_labels = trainSet.getDistinctClassLabels()
        self.__n_classes = len(self.__class_labels)
        self.__training_history = []
        self.__trees = []  # Initialize/reset trees
        
        # Set random seed for reproducibility
        random.seed(parameters.getSeed())
        
        if self.__n_classes == 2:
            # Binary classification
            self.__trainBinary(trainSet, parameters, validationSet)
        else:
            # Multiclass classification
            self.__trainMulticlass(trainSet, parameters, validationSet)
    
    def __trainBinary(self, trainSet: InstanceList, 
                     parameters: XGBoostParameter,
                     validationSet: Optional[InstanceList] = None) -> None:
        """
        Train for binary classification.
        """
        n_samples = trainSet.size()
        
        # Initialize with log odds
        positive_count = sum(1 for i in range(n_samples) 
                            if trainSet.get(i).getClassLabel() == self.__class_labels[1])
        
        if positive_count == 0:
            self.__base_score = -5.0
        elif positive_count == n_samples:
            self.__base_score = 5.0
        else:
            self.__base_score = log(positive_count / (n_samples - positive_count))
        
        predictions = [self.__base_score] * n_samples
        
        # Early stopping variables
        best_val_error = float('inf')
        rounds_without_improvement = 0
        best_n_trees = 0
        
        # Boosting iterations
        for iteration in range(parameters.getNEstimators()):
            # Sample instances
            if parameters.getSubsample() < 1.0:
                n_subsample = max(1, int(n_samples * parameters.getSubsample()))
                sample_indices = random.sample(range(n_samples), n_subsample)
            else:
                sample_indices = list(range(n_samples))
            
            # Calculate gradients and hessians
            gradients = [0.0] * n_samples
            hessians = [0.0] * n_samples
            
            for i in range(n_samples):
                pred_prob = self.__sigmoid(predictions[i])
                true_label = 1.0 if trainSet.get(i).getClassLabel() == self.__class_labels[1] else 0.0
                
                gradients[i] = pred_prob - true_label
                hessians[i] = max(pred_prob * (1.0 - pred_prob), 1e-6)
            
            # Build tree
            tree = XGBoostTree(trainSet, gradients, hessians, sample_indices, parameters)
            self.__trees.append(tree)
            
            # Update predictions
            learning_rate = parameters.getLearningRate()
            for i in range(n_samples):
                predictions[i] += learning_rate * tree.predictValue(trainSet.get(i))
            
            # Early stopping check
            if validationSet is not None:
                val_error = self.__calculateError(validationSet)
                self.__training_history.append({
                    'iteration': iteration,
                    'validation_error': val_error
                })
                
                if val_error < best_val_error:
                    best_val_error = val_error
                    best_n_trees = iteration + 1
                    rounds_without_improvement = 0
                else:
                    rounds_without_improvement += 1
                    
                    if rounds_without_improvement >= parameters.getEarlyStoppingRounds():
                        self.__trees = self.__trees[:best_n_trees]
                        break
    
    def __trainMulticlass(self, trainSet: InstanceList, 
                         parameters: XGBoostParameter,
                         validationSet: Optional[InstanceList] = None) -> None:
        """
        Train for multiclass classification.
        """
        n_samples = trainSet.size()
        
        # Initialize predictions for each class
        predictions = [[0.0 for _ in range(n_samples)] for _ in range(self.__n_classes)]
        
        # Initialize trees as list of lists
        self.__trees = [[] for _ in range(self.__n_classes)]
        
        # Early stopping variables
        best_val_error = float('inf')
        rounds_without_improvement = 0
        best_n_trees = 0
        
        for iteration in range(parameters.getNEstimators()):
            # Sample instances
            if parameters.getSubsample() < 1.0:
                n_subsample = max(1, int(n_samples * parameters.getSubsample()))
                sample_indices = random.sample(range(n_samples), n_subsample)
            else:
                sample_indices = list(range(n_samples))
            
            for class_idx in range(self.__n_classes):
                target_class = self.__class_labels[class_idx]
                
                # Calculate gradients and hessians
                gradients = [0.0] * n_samples
                hessians = [0.0] * n_samples
                
                for i in range(n_samples):
                    scores = [predictions[c][i] for c in range(self.__n_classes)]
                    probs = self.__softmax(scores)
                    
                    true_label = 1.0 if trainSet.get(i).getClassLabel() == target_class else 0.0
                    pred_prob = probs[class_idx]
                    
                    gradients[i] = pred_prob - true_label
                    hessians[i] = max(pred_prob * (1.0 - pred_prob), 1e-6)
                
                # Build tree
                tree = XGBoostTree(trainSet, gradients, hessians, sample_indices, parameters)
                self.__trees[class_idx].append(tree)
                
                # Update predictions
                learning_rate = parameters.getLearningRate()
                for i in range(n_samples):
                    predictions[class_idx][i] += learning_rate * tree.predictValue(trainSet.get(i))
            
            # Early stopping check
            if validationSet is not None:
                val_error = self.__calculateError(validationSet)
                self.__training_history.append({
                    'iteration': iteration,
                    'validation_error': val_error
                })
                
                if val_error < best_val_error:
                    best_val_error = val_error
                    best_n_trees = iteration + 1
                    rounds_without_improvement = 0
                else:
                    rounds_without_improvement += 1
                    
                    if rounds_without_improvement >= parameters.getEarlyStoppingRounds():
                        for class_idx in range(self.__n_classes):
                            self.__trees[class_idx] = self.__trees[class_idx][:best_n_trees]
                        break
    
    def __calculateError(self, testSet: InstanceList) -> float:
        """
        Calculate classification error.
        """
        n_errors = 0
        for i in range(testSet.size()):
            instance = testSet.get(i)
            predicted = self.predict(instance)
            if predicted != instance.getClassLabel():
                n_errors += 1
        return n_errors / testSet.size() if testSet.size() > 0 else 0.0
    
    def predict(self, instance: Instance) -> str:
        """
        Predict class label.
        """
        # Check if multiclass (list of lists) or binary (flat list)
        if self.__trees and isinstance(self.__trees[0], list):
            # Multiclass
            scores = [0.0] * self.__n_classes
            learning_rate = self.__parameter.getLearningRate()
            
            for class_idx in range(self.__n_classes):
                for tree in self.__trees[class_idx]:
                    scores[class_idx] += learning_rate * tree.predictValue(instance)
            
            max_idx = scores.index(max(scores))
            return self.__class_labels[max_idx]
        else:
            # Binary
            score = self.__base_score
            learning_rate = self.__parameter.getLearningRate()
            
            for tree in self.__trees:
                score += learning_rate * tree.predictValue(instance)
            
            prob = self.__sigmoid(score)
            return self.__class_labels[1] if prob >= 0.5 else self.__class_labels[0]
    
    def predictProbability(self, instance: Instance) -> Dict[str, float]:
        """
        Predict probability distribution.
        """
        if self.__trees and isinstance(self.__trees[0], list):
            # Multiclass
            scores = [0.0] * self.__n_classes
            learning_rate = self.__parameter.getLearningRate()
            
            for class_idx in range(self.__n_classes):
                for tree in self.__trees[class_idx]:
                    scores[class_idx] += learning_rate * tree.predictValue(instance)
            
            probs = self.__softmax(scores)
            return {self.__class_labels[i]: probs[i] for i in range(self.__n_classes)}
        else:
            # Binary
            score = self.__base_score
            learning_rate = self.__parameter.getLearningRate()
            
            for tree in self.__trees:
                score += learning_rate * tree.predictValue(instance)
            
            prob_positive = self.__sigmoid(score)
            return {
                self.__class_labels[0]: 1.0 - prob_positive,
                self.__class_labels[1]: prob_positive
            }
    
    def getTrainingHistory(self) -> List[Dict]:
        """
        Get training history.
        """
        return self.__training_history
    
    def getFeatureImportance(self) -> Dict[int, float]:
        """
        Get feature importance scores.
        """
        return self.__feature_importance
    
    def loadModel(self, fileName: str) -> None:
        """
        Load model from file.
        """
        import pickle
        try:
            with open(fileName, 'rb') as f:
                model_data = pickle.load(f)
                self.__trees = model_data['trees']
                self.__class_labels = model_data['class_labels']
                self.__n_classes = model_data['n_classes']
                self.__base_score = model_data['base_score']
                self.__parameter = model_data['parameter']
        except Exception as e:
            raise IOError(f"Failed to load model from {fileName}: {str(e)}")
    
    def saveModel(self, fileName: str) -> None:
        """
        Save model to file.
        """
        import pickle
        try:
            model_data = {
                'trees': self.__trees,
                'class_labels': self.__class_labels,
                'n_classes': self.__n_classes,
                'base_score': self.__base_score,
                'parameter': self.__parameter
            }
            with open(fileName, 'wb') as f:
                pickle.dump(model_data, f)
        except Exception as e:
            raise IOError(f"Failed to save model to {fileName}: {str(e)}")