import numpy as np
import copy

# Implement Greedy Algorithm for split finding in a regression tree
class RegressionTree:
    def __init__(self, n_estimator=1, max_depth=3, reg_lambda=1.0, prune_gamma=0.0):
        self.n_estimator = n_estimator        # Number of estimators
        self.max_depth = max_depth            # Maximum depth of the tree
        self.reg_lambda = reg_lambda          # Regularization constant
        self.prune_gamma = prune_gamma        # Pruning threshold
        self.estimator1 = None                # Tree structure before assigning leaf values
        self.estimator2 = None                # Tree structure with leaf values
        self.feature = None                   # Feature matrix (X)
        self.residual = None                  # Residuals (y - prediction)
        self.base_score = None                # Initial prediction (mean of residuals)

    # Regularized learning objective:
    # Split a node into left and right to maximize gain
    def node_split(self, did):
        r = self.reg_lambda
        max_gain = -np.inf
        d = self.feature.shape[1]            # Number of features

        # Calculate gradient before split
        G = -self.residual[did].sum()        # Sum of residuals
        H = did.shape[0]                     # Number of samples in node
        p_score = (G**2) / (H + r)          # Score before the split

        best_split = None

        # Iterate over all features to find best split
        for k in range(d):
            X_feat = self.feature[did, k]
            x_uniq = np.unique(X_feat)
            # Candidate split points: midpoints between unique sorted feature values
            s_point = [(x_uniq[i-1] + x_uniq[i])/2 for i in range(1, len(x_uniq))]

            l_bound = -np.inf
            for j in s_point:
                # Split samples into left and right nodes
                left = did[(X_feat > l_bound) & (X_feat <= j)]
                right = did[X_feat > j]

                if len(left) == 0 or len(right) == 0:
                    continue

                # Calculate gradients and hessians for left and right
                GL = -self.residual[left].sum()
                HL = left.shape[0]
                GR = G - GL
                HR = H - HL

                # Calculate gain for this split
                gain = (GL**2)/(HL + r) + (GR**2)/(HR + r) - p_score

                if gain > max_gain:
                    max_gain = gain
                    best_split = {"fid": k, "split_point": j, "left": left, "right": right}

                l_bound = j

        # Only split if gain exceeds pruning threshold
        if max_gain >= self.prune_gamma:
            return best_split
        return np.nan  # No valid split found

    # Recursively split tree nodes until maximum depth is reached
    def recursive_split(self, node, curr_depth):
        if curr_depth >= self.max_depth or not isinstance(node, dict):
            return

        self.recursive_split(node.get("left"), curr_depth + 1)
        self.recursive_split(node.get("right"), curr_depth + 1)

    # Calculate output value for a leaf node (regularized)
    def output_value(self, did):
        return np.sum(self.residual[did]) / (did.shape[0] + self.reg_lambda)

    # Assign output values to all leaf nodes in the tree
    def output_leaf(self, d):
        if isinstance(d, dict):
            for key in ["left", "right"]:
                val = d[key]
                if isinstance(val, dict):
                    self.output_leaf(val)
                else:
                    # Replace node indices with actual leaf value
                    d[key] = self.output_value(val)

    # Fit the regression tree to feature matrix X and residuals y
    def fit(self, x, y):
        self.feature = x
        self.residual = y
        self.base_score = y.mean()  # Initial prediction (mean of residuals)

        # Build the tree from root
        root = self.node_split(np.arange(x.shape[0]))
        if isinstance(root, dict):
            self.recursive_split(root, curr_depth=1)
            self.estimator1 = root
            self.estimator2 = copy.deepcopy(root)
            self.output_leaf(self.estimator2)  # Assign leaf values

        return self.estimator2

    # Predict output for a single sample
    def x_predict(self, p, x):
        if x[p["fid"]] <= p["split_point"]:
            if isinstance(p["left"], dict):
                return self.x_predict(p["left"], x)
            else:
                return p["left"]
        else:
            if isinstance(p["right"], dict):
                return self.x_predict(p["right"], x)
            else:
                return p["right"]

    # Predict outputs for multiple samples
    def predict(self, x_test):
        if self.estimator2 is None:
            # If tree is empty, return base score
            return np.array([self.base_score] * x_test.shape[0])
        # Traverse tree for each sample
        return np.array([self.x_predict(self.estimator2, x) for x in x_test])
 #Built XGBoostRegressor
class MyXGBoostRegressor:
    def __init__(self, n_estimators=10, max_depth=3, reg_lambda=1.0, prune_gamma=0.0, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.prune_gamma = prune_gamma
        self.learning_rate = learning_rate
        self.trees = []
        self.base_score = None

    def fit(self, X, y):
        n_samples = X.shape[0]
        # Initial prediction: mean of y
        self.base_score = y.mean()
        y_pred = np.full(n_samples, self.base_score)

        for m in range(self.n_estimators):
            # Compute residuals (negative gradient for squared error)
            residuals = y - y_pred
            tree = RegressionTree(max_depth=self.max_depth, reg_lambda=self.reg_lambda, prune_gamma=self.prune_gamma)
            tree.fit(X, residuals)
            # Predict residuals and update y_pred
            update = tree.predict(X)
            y_pred += self.learning_rate * update
            self.trees.append(tree)

    def predict(self, X):
        y_pred = np.full(X.shape[0], self.base_score)
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred