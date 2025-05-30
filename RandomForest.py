import numpy as np
from collections import Counter
import pandas as pd
from RandomForestParams import FIXED_PARAMS
from DecisionTree import DecisionTree
import time

def segmenta_dataset(x, y, test_size=0.2, random_state=None):
    if random_state:
        np.random.seed(random_state)

    n_samples = x.shape[0]
    test_samples = int(n_samples * test_size)
    indices = np.random.permutation(n_samples)
    test_idx = indices[:test_samples]
    train_idx = indices[test_samples:]
    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]

def taxa_acertos(y_true, y_pred):
    acertos = np.sum(y_true == y_pred)
    return acertos / len(y_true)

class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_feature=None):
        self.n_trees = n_trees
        self.max_depth=max_depth
        self.min_samples_split=min_samples_split
        self.n_features=n_feature
        self.trees = []
        
    def fit(self, x, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth,
                            min_samples_split=self.min_samples_split,
                            n_features=self.n_features)
            x_sample, y_sample = self._bootstrap_samples(x, y)
            tree.fit(x_sample, y_sample)
            self.trees.append(tree)
    
    def _bootstrap_samples(self, x, y):
        n_samples = x.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return x[idxs], y[idxs]
    
    def _voto_majoritario(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]
    
    def predict(self, x):
        predictions = np.array([tree.predict(x) for tree in self.trees])
        tree_preds = np.swapaxes(predictions, 0, 1)
        return np.array([self._voto_majoritario(pred) for pred in tree_preds])

if __name__ == "__main__":
    start_time = time.time()
    data = np.genfromtxt('treino_sinais_vitais_com_label.txt', delimiter=',', skip_header=1)
    x = data[:, :-2]
    y = data[:, -1].astype(int)

    x_train, x_test, y_train, y_test = segmenta_dataset(x, y, test_size=0.2)
    clf = RandomForest(**FIXED_PARAMS)
    clf.fit(x_train, y_train)
    print(f"Time Spent Training: {time.time() - start_time}s")

    start_time = time.time()
    predictions = clf.predict(x_test)
    print(f'Accuracy: {taxa_acertos(y_test, predictions)}')
    print(f"Time Spent Predicting: {time.time() - start_time}s")