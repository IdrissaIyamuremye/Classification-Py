import numpy as np
import copy


# Classification Tree (single tree)

class ClassificationTree:
    def __init__(self, max_depth=3, reg_lambda=1.0, prune_gamma=0.0):
        self.max_depth = max_depth        # Maximum depth
        self.reg_lambda = reg_lambda      # Regularization constant
        self.prune_gamma = prune_gamma    # Pruning threshold
        self.estimator1 = None            # Tree structure before assigning leaf values
        self.estimator2 = None            # Tree structure with leaf values
        self.feature = None               # Feature matrix
        self.residual = None              # Residuals (negative gradient)
        self.base_score = None            # Initial log-odds

    # Find best split for current node
    def node_split(self, did):
        r = self.reg_lambda
        max_gain = -np.inf
        d = self.feature.shape[1]
        G = -self.residual[did].sum()
        H = did.shape[0]
        p_score = (G**2)/(H + r)
        best_split = None

        for k in range(d):
            X_feat = self.feature[did, k]
            x_uniq = np.unique(X_feat)
            s_point = [(x_uniq[i-1]+x_uniq[i])/2 for i in range(1,len(x_uniq))]
            l_bound = -np.inf
            for j in s_point:
                left = did[(X_feat>l_bound)&(X_feat<=j)]
                right = did[X_feat>j]
                if len(left)==0 or len(right)==0:
                    continue
                GL = -self.residual[left].sum()
                HL = left.shape[0]
                GR = G - GL
                HR = H - HL
                gain = (GL**2)/(HL+r) + (GR**2)/(HR+r) - p_score
                if gain > max_gain:
                    max_gain = gain
                    best_split = {"fid": k, "split_point": j, "left": left, "right": right}
                l_bound = j
        if max_gain >= self.prune_gamma:
            return best_split
        return np.nan

    # Recursively split nodes
    def recursive_split(self, node, curr_depth):
        if curr_depth >= self.max_depth or not isinstance(node, dict):
            return
        self.recursive_split(node.get("left"), curr_depth+1)
        self.recursive_split(node.get("right"), curr_depth+1)

    # Leaf value for log-loss
    def output_value(self, did):
        return np.sum(self.residual[did]) / (did.shape[0] + self.reg_lambda)

    # Assign leaf values to all leaves
    def output_leaf(self, d):
        if isinstance(d, dict):
            for key in ["left","right"]:
                val = d[key]
                if isinstance(val, dict):
                    self.output_leaf(val)
                else:
                    d[key] = self.output_value(val)

    # Fit tree to residuals
    def fit(self, X, residuals):
        self.feature = X
        self.residual = residuals
        root = self.node_split(np.arange(X.shape[0]))
        if isinstance(root, dict):
            self.recursive_split(root, curr_depth=1)
            self.estimator2 = copy.deepcopy(root)
            self.output_leaf(self.estimator2)
        return self.estimator2

    # Predict single sample
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

    # Predict multiple samples
    def predict(self, X):
        if self.estimator2 is None:
            return np.zeros(X.shape[0])
        return np.array([self.x_predict(self.estimator2, x) for x in X])

# Built  XGBoost Classifier
class MyXGBoostClassifier:
    def __init__(self, n_estimator=10, max_depth=3, reg_lambda=1.0, prune_gamma=0.0, learning_rate=0.1):
        self.n_estimator = n_estimator        # Number of trees
        self.max_depth = max_depth            # Maximum depth of each tree
        self.reg_lambda = reg_lambda          # Regularization constant
        self.prune_gamma = prune_gamma        # Pruning threshold
        self.learning_rate = learning_rate    # Learning rate
        self.trees = []                       # List to store trees
        self.base_score = None                # Initial log-odds

    # Sigmoid to convert log-odds to probability
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    # Fit ensemble
    def fit(self, X, y):
        n_samples = X.shape[0]
        # Initial log-odds
        p = np.clip(np.mean(y),1e-6,1-1e-6)
        self.base_score = np.log(p/(1-p))
        y_pred = np.full(n_samples, self.base_score)

        for m in range(self.n_estimator):
            # Compute residuals: negative gradient of log-loss
            p_pred = self.sigmoid(y_pred)
            residuals = y - p_pred
            tree = ClassificationTree(max_depth=self.max_depth,
                                      reg_lambda=self.reg_lambda,
                                      prune_gamma=self.prune_gamma)
            tree.fit(X, residuals)
            update = tree.predict(X)
            y_pred += self.learning_rate * update
            self.trees.append(tree)

    # Predict probability
    def predict_proba(self, X):
        y_pred = np.full(X.shape[0], self.base_score)
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)
        return self.sigmoid(y_pred)

    # Predict class label
    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
