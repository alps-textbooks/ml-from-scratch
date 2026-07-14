"""Classification tree built from scratch: a decision tree classifier plus the
`penguins` dataset train/test split.

This is an importable module extracted from ``classification_tree.ipynb`` so
that sibling notebooks (e.g. boosting) can do ``import classification_tree as
ct`` and reference ``ct.all_rows_equal`` / ``ct.possible_splits`` (and
``ct.DecisionTreeClassifier``) without relying on ``import_ipynb``. The dataset
is read from a local ``penguins.csv`` rather than a remote URL so the module
imports in sandboxed (in-browser) runtimes with no network access.
"""

import pandas as pd
import numpy as np
from itertools import combinations

## Load data
penguins = pd.read_csv('penguins.csv')
penguins.dropna(inplace = True)
X = np.array(penguins.drop(columns = ['species','island']))
y = np.array(penguins['species'])

## Train-test split
np.random.seed(1)
test_frac = 0.25
test_size = int(len(y)*test_frac)
test_idxs = np.random.choice(np.arange(len(y)), test_size, replace = False)
X_train = np.delete(X, test_idxs, 0)
y_train = np.delete(y, test_idxs, 0)
X_test = X[test_idxs]
y_test = y[test_idxs]


## Loss Functions
def gini_index(y):
    size = len(y)
    classes, counts = np.unique(y, return_counts = True)
    pmk = counts/size
    return np.sum(pmk*(1-pmk))

def cross_entropy(y):
    size = len(y)
    classes, counts = np.unique(y, return_counts = True)
    pmk = counts/size
    return -np.sum(pmk*np.log2(pmk))

def split_loss(child1, child2, loss = cross_entropy):
    return (len(child1)*loss(child1) + len(child2)*loss(child2))/(len(child1) + len(child2))


## Helper Functions
def all_rows_equal(X):
    return (X == X[0]).all()

def possible_splits(x):
    L_values = []
    for i in range(1, int(np.floor(len(x)/2)) + 1):
        L_values.extend(list(combinations(x, i)))
    return L_values


## Helper Classes
class Node:

    def __init__(self, Xsub, ysub, ID, depth = 0, parent_ID = None, leaf = True):
        self.ID = ID
        self.Xsub = Xsub
        self.ysub = ysub
        self.size = len(ysub)
        self.depth = depth
        self.parent_ID = parent_ID
        self.leaf = leaf


class Splitter:

    def __init__(self):
        self.loss = np.inf
        self.no_split = True

    def _replace_split(self, loss, d, dtype = 'quant', t = None, L_values = None):
        self.loss = loss
        self.d = d
        self.dtype = dtype
        self.t = t
        self.L_values = L_values
        self.no_split = False


## Main Class
class DecisionTreeClassifier:

    #############################
    ######## 1. TRAINING ########
    #############################

    ######### FIT ##########
    def fit(self, X, y, loss_func = cross_entropy, max_depth = 100, min_size = 2, C = None):

        ## Add data
        self.X = X
        self.y = y
        self.N, self.D = self.X.shape
        dtypes = [np.array(list(self.X[:,d])).dtype for d in range(self.D)]
        self.dtypes = ['quant' if (dtype == float or dtype == int) else 'cat' for dtype in dtypes]

        ## Add model parameters
        self.loss_func = loss_func
        self.max_depth = max_depth
        self.min_size = min_size
        self.C = C

        ## Initialize nodes
        self.nodes_dict = {}
        self.current_ID = 0
        initial_node = Node(Xsub = X, ysub = y, ID = self.current_ID, parent_ID = None)
        self.nodes_dict[self.current_ID] = initial_node
        self.current_ID += 1

        # Build
        self._build()

    ###### BUILD TREE ######
    def _build(self):

        eligible_buds = self.nodes_dict
        for layer in range(self.max_depth):

            ## Find eligible nodes for layer iteration
            eligible_buds = {ID:node for (ID, node) in self.nodes_dict.items() if
                                (node.leaf == True) &
                                (node.size >= self.min_size) &
                                (~all_rows_equal(node.Xsub)) &
                                (len(np.unique(node.ysub)) > 1)}
            if len(eligible_buds) == 0:
                break

            ## split each eligible parent
            for ID, bud in eligible_buds.items():

                ## Find split
                self._find_split(bud)

                ## Make split
                if not self.splitter.no_split:
                    self._make_split()

    ###### FIND SPLIT ######
    def _find_split(self, bud):

        ## Instantiate splitter
        splitter = Splitter()
        splitter.bud_ID = bud.ID

        ## For each (eligible) predictor...
        if self.C is None:
            eligible_predictors = np.arange(self.D)
        else:
            eligible_predictors = np.random.choice(np.arange(self.D), self.C, replace = False)
        for d in sorted(eligible_predictors):
            Xsub_d = bud.Xsub[:,d]
            dtype = self.dtypes[d]
            if len(np.unique(Xsub_d)) == 1:
                continue

            ## For each value...
            if dtype == 'quant':
                for t in np.unique(Xsub_d)[:-1]:
                    ysub_L = bud.ysub[Xsub_d <= t]
                    ysub_R = bud.ysub[Xsub_d > t]
                    loss = split_loss(ysub_L, ysub_R, loss = self.loss_func)
                    if loss < splitter.loss:
                        splitter._replace_split(loss, d, 'quant', t = t)
            else:
                for L_values in possible_splits(np.unique(Xsub_d)):
                    ysub_L = bud.ysub[np.isin(Xsub_d, L_values)]
                    ysub_R = bud.ysub[~np.isin(Xsub_d, L_values)]
                    loss = split_loss(ysub_L, ysub_R, loss = self.loss_func)
                    if loss < splitter.loss:
                        splitter._replace_split(loss, d, 'cat', L_values = L_values)

        ## Save splitter
        self.splitter = splitter

    ###### MAKE SPLIT ######
    def _make_split(self):

        ## Update parent node
        parent_node = self.nodes_dict[self.splitter.bud_ID]
        parent_node.leaf = False
        parent_node.child_L = self.current_ID
        parent_node.child_R = self.current_ID + 1
        parent_node.d = self.splitter.d
        parent_node.dtype = self.splitter.dtype
        parent_node.t = self.splitter.t
        parent_node.L_values = self.splitter.L_values

        ## Get X and y data for children
        if parent_node.dtype == 'quant':
            L_condition = parent_node.Xsub[:,parent_node.d] <= parent_node.t

        else:
            L_condition = np.isin(parent_node.Xsub[:,parent_node.d], parent_node.L_values)
        Xchild_L = parent_node.Xsub[L_condition]
        ychild_L = parent_node.ysub[L_condition]
        Xchild_R = parent_node.Xsub[~L_condition]
        ychild_R = parent_node.ysub[~L_condition]

        ## Create child nodes
        child_node_L = Node(Xchild_L, ychild_L, depth = parent_node.depth + 1,
                            ID = self.current_ID, parent_ID = parent_node.ID)
        child_node_R = Node(Xchild_R, ychild_R, depth = parent_node.depth + 1,
                            ID = self.current_ID+1, parent_ID = parent_node.ID)
        self.nodes_dict[self.current_ID] = child_node_L
        self.nodes_dict[self.current_ID + 1] = child_node_R
        self.current_ID += 2


    #############################
    ####### 2. PREDICTING #######
    #############################

    ###### LEAF MODES ######
    def _get_leaf_modes(self):
        self.leaf_modes = {}
        for node_ID, node in self.nodes_dict.items():
            if node.leaf:
                values, counts = np.unique(node.ysub, return_counts=True)
                self.leaf_modes[node_ID] = values[np.argmax(counts)]

    ####### PREDICT ########
    def predict(self, X_test):

        # Calculate leaf modes
        self._get_leaf_modes()

        yhat = []
        for x in X_test:
            node = self.nodes_dict[0]
            while not node.leaf:
                if node.dtype == 'quant':
                    if x[node.d] <= node.t:
                        node = self.nodes_dict[node.child_L]
                    else:
                        node = self.nodes_dict[node.child_R]
                else:
                    if x[node.d] in node.L_values:
                        node = self.nodes_dict[node.child_L]
                    else:
                        node = self.nodes_dict[node.child_R]
            yhat.append(self.leaf_modes[node.ID])
        return np.array(yhat)
