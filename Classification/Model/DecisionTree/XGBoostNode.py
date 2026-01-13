"""
XGBoost Decision Node
"""

from typing import List, Optional
from Classification.Instance.Instance import Instance
from Classification.InstanceList.InstanceList import InstanceList
from Classification.Model.DecisionTree.DecisionNode import DecisionNode
from Classification.Parameter.XGBoostParameter import XGBoostParameter


class XGBoostNode(DecisionNode):
    """
    Extended DecisionNode for XGBoost that supports gradient-based splits.
    Implements XGBoost-specific splitting logic with proper regularization.
    
    Attributes:
        EPSILON (float): Numerical stability constant
        __gradients (List[float]): Gradient values for instances
        __hessians (List[float]): Hessian values for instances
        __instance_indices (List[int]): Indices of instances in this node
        __parameter (XGBoostParameter): XGBoost hyperparameters
        __depth (int): Current depth of the node in the tree
        __leaf_value (float): Predicted value for leaf nodes
        __feature_subset (Optional[List[int]]): Subset of features to consider for splitting
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
        
        Args:
            data (InstanceList): Training instances
            gradients (List[float]): Gradient values
            hessians (List[float]): Hessian values
            instance_indices (List[int]): Indices of instances in this node
            condition: Split condition for this node
            parameter (XGBoostParameter): Hyperparameters
            depth (int): Current tree depth
            feature_subset (Optional[List[int]]): Features to consider
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
        
        Args:
            data (InstanceList): Training instances
            gradients (List[float]): Gradient values
            hessians (List[float]): Hessian values
            instance_indices (List[int]): Indices of instances to split
            parameter (XGBoostParameter): Hyperparameters
            
        Returns:
            None: Updates node children if beneficial split found
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
        
        Args:
            instance (Instance): Instance to predict
            
        Returns:
            float: Predicted leaf value (weight) for this instance
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