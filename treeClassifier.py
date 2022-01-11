#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

#Load data
X_train = np.loadtxt('data/X_train.csv', delimiter=",")
y_train = np.loadtxt('data/y_train.csv', delimiter=",").astype(int)
X_test = np.loadtxt('data/X_test.csv', delimiter=",")
y_test = np.loadtxt('data/y_test.csv', delimiter=",").astype(int)

#Get the same result every time
np.random.seed(0)

class Node:
    def __init__(self, predicted_class):
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None


class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y, loss_type):
        self.n_classes_ = len(set(y))
        self.n_features_ = X.shape[1]
        self.tree_ = self._grow_tree(X, y, loss_type)
        self.loss_type = loss_type
        
    def predict(self, X):
        return [self._predict(inputs) for inputs in X]
    

    def _best_split(self, X, y, loss_type):
        m = y.size
        if m <= 1:
            return None, None
        num_parent = [np.sum(y == c) for c in range(self.n_classes_)]
        if loss_type == 'gini':
            best_loss = sum((n / m) *(1-(n/m)) for n in num_parent)
        if loss_type == 'entropy':
            best_loss = sum(-(n/m)*np.log2(n/m) -(1-(n/m))*np.log2(1-(n/m)) for n in num_parent)
        if loss_type == 'misclassification':
            best_loss = np.min([(n/m) for n in num_parent])
        best_idx, best_thr = None, None
        for idx in range(self.n_features_):
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))
            num_left = [0] * self.n_classes_
            num_right = num_parent.copy()
            for i in range(1, m):
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1
                
                if loss_type == 'gini':
                    loss_left = sum((num_left[x] / i) *(1-(num_left[x] / i)) for x in range(self.n_classes_))
                    loss_right = sum((num_right[x] / (m - i)) *(1 - (num_right[x] / (m - i))) for x in range(self.n_classes_))
                     
                if loss_type == 'entropy':
                    loss_left = sum(-(num_left[x] / i)*np.log2((num_left[x] / i)) -(1-(num_left[x] / i))*np.log2(1-(num_left[x] / i))
                                    for x in range(self.n_classes_))
                    loss_right = sum(-(num_right[x] / (m - i))*np.log2((num_right[x] / (m - i))) -(1-(num_right[x] / (m - i)))*np.log2(1-(num_right[x] / (m - i))) for x in range(self.n_classes_))
                    
                if loss_type == 'misclassification':
                    loss_left = np.min([(num_left[x] / i) for x in range(self.n_classes_)])
                    loss_right = np.min([(num_right[x] / (m - i)) for x in range(self.n_classes_)])
                
                loss = (i * loss_left + (m - i) * loss_right) / m
                if thresholds[i] == thresholds[i - 1]:
                    continue
                if loss < best_loss:
                    best_loss = loss
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2
        return best_idx, best_thr

    def _grow_tree(self, X, y, loss_type, depth=0):
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes_)]
        predicted_class = np.argmax(num_samples_per_class)
        node = Node(predicted_class=predicted_class)
        if depth < self.max_depth:
            idx, thr = self._best_split(X, y, loss_type)
            if idx is not None:
                indices_left = X[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node.feature_index = idx
                node.threshold = thr
                node.left = self._grow_tree(X_left, y_left, loss_type, depth + 1)
                node.right = self._grow_tree(X_right, y_right, loss_type, depth + 1)
        return node

    def _predict(self, inputs):
        node = self.tree_
        while node.left:
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


tree_0 = DecisionTreeClassifier(max_depth = 0)
tree_1 = DecisionTreeClassifier(max_depth = 1)
tree_2 = DecisionTreeClassifier(max_depth = 2)
tree_3 = DecisionTreeClassifier(max_depth = 3)
tree_4 = DecisionTreeClassifier(max_depth = 4)
tree_5 = DecisionTreeClassifier(max_depth = 5)
tree_6 = DecisionTreeClassifier(max_depth = 6)
tree_7 = DecisionTreeClassifier(max_depth = 7)
tree_8 = DecisionTreeClassifier(max_depth = 8)



#Gini Models with different max_depths from 0 to 8
tree_0.fit(X_train, y_train, loss_type = 'gini')
tree_1.fit(X_train, y_train, loss_type = 'gini')
tree_2.fit(X_train, y_train, loss_type = 'gini')
tree_3.fit(X_train, y_train, loss_type = 'gini')
tree_4.fit(X_train, y_train, loss_type = 'gini')
tree_5.fit(X_train, y_train, loss_type = 'gini')
tree_6.fit(X_train, y_train, loss_type = 'gini')
tree_7.fit(X_train, y_train, loss_type = 'gini')
tree_8.fit(X_train, y_train, loss_type = 'gini')

max_depth = [0,1,2,3,4,5,6,7,8]
gini_training_accuracy = []
gini_training_accuracy.append(accuracy_metric(y_train, tree_0.predict(X_train)))
gini_training_accuracy.append(accuracy_metric(y_train, tree_1.predict(X_train)))
gini_training_accuracy.append(accuracy_metric(y_train, tree_2.predict(X_train)))
gini_training_accuracy.append(accuracy_metric(y_train, tree_3.predict(X_train)))
gini_training_accuracy.append(accuracy_metric(y_train, tree_4.predict(X_train)))
gini_training_accuracy.append(accuracy_metric(y_train, tree_5.predict(X_train)))
gini_training_accuracy.append(accuracy_metric(y_train, tree_6.predict(X_train)))
gini_training_accuracy.append(accuracy_metric(y_train, tree_7.predict(X_train)))
gini_training_accuracy.append(accuracy_metric(y_train, tree_8.predict(X_train)))

gini_test_accuracy = []
gini_test_accuracy.append(accuracy_metric(y_test, tree_0.predict(X_test)))
gini_test_accuracy.append(accuracy_metric(y_test, tree_1.predict(X_test)))
gini_test_accuracy.append(accuracy_metric(y_test, tree_2.predict(X_test)))
gini_test_accuracy.append(accuracy_metric(y_test, tree_3.predict(X_test)))
gini_test_accuracy.append(accuracy_metric(y_test, tree_4.predict(X_test)))
gini_test_accuracy.append(accuracy_metric(y_test, tree_5.predict(X_test)))
gini_test_accuracy.append(accuracy_metric(y_test, tree_6.predict(X_test)))
gini_test_accuracy.append(accuracy_metric(y_test, tree_7.predict(X_test)))
gini_test_accuracy.append(accuracy_metric(y_test, tree_8.predict(X_test)))


plt.figure(figsize = (10,8))
plt.plot(max_depth, gini_training_accuracy)
plt.plot(max_depth, gini_test_accuracy)
plt.title('Training and Test Accuracy on Decision Trees Using Gini Loss')
plt.legend(['Train Accuracy', 'Test Accuracy'])
plt.xlabel('Max_Depth')
plt.ylabel('Accuracy %')


print(gini_training_accuracy)
print(gini_test_accuracy)


tree_0.fit(X_train, y_train, loss_type = 'entropy')
tree_1.fit(X_train, y_train, loss_type = 'entropy')
tree_2.fit(X_train, y_train, loss_type = 'entropy')
tree_3.fit(X_train, y_train, loss_type = 'entropy')
tree_4.fit(X_train, y_train, loss_type = 'entropy')
tree_5.fit(X_train, y_train, loss_type = 'entropy')
tree_6.fit(X_train, y_train, loss_type = 'entropy')
tree_7.fit(X_train, y_train, loss_type = 'entropy')
tree_8.fit(X_train, y_train, loss_type = 'entropy')

entropy_training_accuracy = []
entropy_training_accuracy.append(accuracy_metric(y_train, tree_0.predict(X_train)))
entropy_training_accuracy.append(accuracy_metric(y_train, tree_1.predict(X_train)))
entropy_training_accuracy.append(accuracy_metric(y_train, tree_2.predict(X_train)))
entropy_training_accuracy.append(accuracy_metric(y_train, tree_3.predict(X_train)))
entropy_training_accuracy.append(accuracy_metric(y_train, tree_4.predict(X_train)))
entropy_training_accuracy.append(accuracy_metric(y_train, tree_5.predict(X_train)))
entropy_training_accuracy.append(accuracy_metric(y_train, tree_6.predict(X_train)))
entropy_training_accuracy.append(accuracy_metric(y_train, tree_7.predict(X_train)))
entropy_training_accuracy.append(accuracy_metric(y_train, tree_8.predict(X_train)))

entropy_test_accuracy = []
entropy_test_accuracy.append(accuracy_metric(y_test, tree_0.predict(X_test)))
entropy_test_accuracy.append(accuracy_metric(y_test, tree_1.predict(X_test)))
entropy_test_accuracy.append(accuracy_metric(y_test, tree_2.predict(X_test)))
entropy_test_accuracy.append(accuracy_metric(y_test, tree_3.predict(X_test)))
entropy_test_accuracy.append(accuracy_metric(y_test, tree_4.predict(X_test)))
entropy_test_accuracy.append(accuracy_metric(y_test, tree_5.predict(X_test)))
entropy_test_accuracy.append(accuracy_metric(y_test, tree_6.predict(X_test)))
entropy_test_accuracy.append(accuracy_metric(y_test, tree_7.predict(X_test)))
entropy_test_accuracy.append(accuracy_metric(y_test, tree_8.predict(X_test)))


plt.figure(figsize = (10,8))
plt.plot(max_depth, entropy_training_accuracy)
plt.plot(max_depth, entropy_test_accuracy)
plt.title('Training and Test Accuracy on Decision Trees Using Entropy Loss')
plt.legend(['Train Accuracy', 'Test Accuracy'])
plt.xlabel('Max_Depth')
plt.ylabel('Accuracy %')


print(entropy_training_accuracy)
print(entropy_test_accuracy)


tree_0.fit(X_train, y_train, loss_type = 'misclassification')
tree_1.fit(X_train, y_train, loss_type = 'misclassification')
tree_2.fit(X_train, y_train, loss_type = 'misclassification')
tree_3.fit(X_train, y_train, loss_type = 'misclassification')
tree_4.fit(X_train, y_train, loss_type = 'misclassification')
tree_5.fit(X_train, y_train, loss_type = 'misclassification')
tree_6.fit(X_train, y_train, loss_type = 'misclassification')
tree_7.fit(X_train, y_train, loss_type = 'misclassification')
tree_8.fit(X_train, y_train, loss_type = 'misclassification')

misclassification_training_accuracy = []
misclassification_training_accuracy.append(accuracy_metric(y_train, tree_0.predict(X_train)))
misclassification_training_accuracy.append(accuracy_metric(y_train, tree_1.predict(X_train)))
misclassification_training_accuracy.append(accuracy_metric(y_train, tree_2.predict(X_train)))
misclassification_training_accuracy.append(accuracy_metric(y_train, tree_3.predict(X_train)))
misclassification_training_accuracy.append(accuracy_metric(y_train, tree_4.predict(X_train)))
misclassification_training_accuracy.append(accuracy_metric(y_train, tree_5.predict(X_train)))
misclassification_training_accuracy.append(accuracy_metric(y_train, tree_6.predict(X_train)))
misclassification_training_accuracy.append(accuracy_metric(y_train, tree_7.predict(X_train)))
misclassification_training_accuracy.append(accuracy_metric(y_train, tree_8.predict(X_train)))

misclassification_test_accuracy = []
misclassification_test_accuracy.append(accuracy_metric(y_test, tree_0.predict(X_test)))
misclassification_test_accuracy.append(accuracy_metric(y_test, tree_1.predict(X_test)))
misclassification_test_accuracy.append(accuracy_metric(y_test, tree_2.predict(X_test)))
misclassification_test_accuracy.append(accuracy_metric(y_test, tree_3.predict(X_test)))
misclassification_test_accuracy.append(accuracy_metric(y_test, tree_4.predict(X_test)))
misclassification_test_accuracy.append(accuracy_metric(y_test, tree_5.predict(X_test)))
misclassification_test_accuracy.append(accuracy_metric(y_test, tree_6.predict(X_test)))
misclassification_test_accuracy.append(accuracy_metric(y_test, tree_7.predict(X_test)))
misclassification_test_accuracy.append(accuracy_metric(y_test, tree_8.predict(X_test)))


print(misclassification_training_accuracy)
print(misclassification_test_accuracy)


plt.figure(figsize = (10,8))
plt.plot(max_depth, misclassification_training_accuracy)
plt.plot(max_depth, misclassification_test_accuracy)
plt.title('Training and Test Accuracy on Decision Trees Using Misclassification Loss')
plt.legend(['Train Accuracy', 'Test Accuracy'])
plt.xlabel('Max_Depth')
plt.ylabel('Accuracy %')


def bagging_accuracy(iterations, trees, X_train, y_train, X_test, y_test):
    size = y_train.size
    division = len(X_train)
    accuracy = []
    for i in range(0,iterations):
        forest_predictions = []
        for i in range(0,trees):
            samples = np.random.choice(X_train.shape[0], division, replace = True)
            X = X_train[samples]
            y = y_train[samples]
            classifier = DecisionTreeClassifier(max_depth = 3)
            classifier.fit(X, y, loss_type = 'entropy')
            forest_predictions.append(classifier.predict(X_test))
        forest_predictions = np.array(forest_predictions).T
        prediction = []
        for i in forest_predictions:
            unique, counts = np.unique(i, return_counts=True)
            values = dict(zip(counts, unique))
            prediction.append(values.get(max(values)))
        accuracy.append(accuracy_metric(y_test, prediction))
    
    median = np.median(accuracy)
    maximum = max(accuracy)
    minimum = min(accuracy)
    return median, maximum, minimum


bagging_accuracy(11, 101, X_train, y_train, X_test, y_test)


#Decide random
class RandomTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y, loss_type):
        self.n_classes_ = len(set(y))
        self.n_features_ = X.shape[1]
        self.tree_ = self._grow_tree(X, y, loss_type)
        self.loss_type = loss_type
        
    def predict(self, X):
        return [self._predict(inputs) for inputs in X]
    

    def _best_split(self, X, y, loss_type):
        m = y.size
        if m <= 1:
            return None, None
        num_parent = [np.sum(y == c) for c in range(self.n_classes_)]
        if loss_type == 'gini':
            best_loss = sum((n / m) *(1-(n/m)) for n in num_parent)
        if loss_type == 'entropy':
            best_loss = sum(-(n/m)*np.log2(n/m) -(1-(n/m))*np.log2(1-(n/m)) for n in num_parent)
        if loss_type == 'misclassification':
            best_loss = np.min([(n/m) for n in num_parent])
        best_idx, best_thr = None, None
        for idx in np.random.permutation(13)[:4]:
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))
            num_left = [0] * self.n_classes_
            num_right = num_parent.copy()
            for i in range(1, m):
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1
                if loss_type == 'gini':
                    loss_left = sum((num_left[x] / i) *(1-(num_left[x] / i)) for x in range(self.n_classes_))
                    #gini_left = 1.0 - sum((num_left[x] / i) ** 2 for x in range(self.n_classes_))
                    loss_right = sum((num_right[x] / (m - i)) *(1 - (num_right[x] / (m - i))) for x in range(self.n_classes_))
                    #gini_right = 1.0 - sum((num_right[x] / (m - i)) ** 2 for x in range(self.n_classes_))
                    
                if loss_type == 'entropy':
                    loss_left = sum(-(num_left[x] / i)*np.log2((num_left[x] / i)) -(1-(num_left[x] / i))*np.log2(1-(num_left[x] / i)) 
                                    for x in range(self.n_classes_))
                    loss_right = sum(-(num_right[x] / (m - i))*np.log2((num_right[x] / (m - i))) -(1-(num_right[x] / (m - i)))*np.log2(1-(num_right[x] / (m - i))) for x in range(self.n_classes_))
                    
                if loss_type == 'misclassification':
                    loss_left = np.min([(num_left[x] / i) for x in range(self.n_classes_)])
                    loss_right = np.min([(num_right[x] / (m - i)) for x in range(self.n_classes_)])
                
                loss = (i * loss_left + (m - i) * loss_right) / m
                if thresholds[i] == thresholds[i - 1]:
                    continue
                if loss < best_loss:
                    best_loss = loss
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2
        return best_idx, best_thr

    def _grow_tree(self, X, y, loss_type, depth=0):
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes_)]
        predicted_class = np.argmax(num_samples_per_class)
        node = Node(predicted_class=predicted_class)
        if depth < self.max_depth:
            idx, thr = self._best_split(X, y, loss_type)
            if idx is not None:
                indices_left = X[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node.feature_index = idx
                node.threshold = thr
                node.left = self._grow_tree(X_left, y_left, loss_type, depth + 1)
                node.right = self._grow_tree(X_right, y_right, loss_type, depth + 1)
        return node

    def _predict(self, inputs):
        node = self.tree_
        while node.left:
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class


def random_accuracy(iterations, trees, X_train, y_train, X_test, y_test):
    size = y_train.size
    division = len(X_train)
    accuracy = []
    for i in range(0,iterations):
        forest_predictions = []
        for i in range(0,trees):
            samples = np.random.choice(X_train.shape[0], division, replace = True)
            X = X_train[samples]
            y = y_train[samples]
            classifier = RandomTreeClassifier(max_depth = 3)
            classifier.fit(X, y, loss_type = 'entropy')
            forest_predictions.append(classifier.predict(X_test))
        forest_predictions = np.array(forest_predictions).T
        prediction = []
        for i in forest_predictions:
            unique, counts = np.unique(i, return_counts=True)
            values = dict(zip(counts, unique))
            prediction.append(values.get(max(values)))
        accuracy.append(accuracy_metric(y_test, prediction))
    
    median = np.median(accuracy)
    maximum = max(accuracy)
    minimum = min(accuracy)
    return maximum, median, minimum


random_accuracy(11, 101, X_train, y_train, X_test, y_test)






