import pandas as pd
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import time


class Node(object):
    def __init__(self):
        self.name = None
        self.node_type = None
        self.predicted_class = None
        self.X = None
        self.test_attribute = None
        self.test_value = None
        self.children = []
    def __repr__(self):
        if self.node_type != 'leaf':
            s = (f"{self.name} Internal node with {self.X.shape[0]} examples, "
                 f"tests attribute {self.test_attribute} at {self.test_value}")
           
        else:
            s = (f"{self.name} Leaf with {self.X.shape[0]} examples, predicts"
                 f" {self.predicted_class}")
        return s


class DecisionTree(object):

    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        
    def fit(self, X, y):
        '''
        Fit a tree on data, in which X is a 2-d numpy array
        of inputs, and y is a 1-d numpy array of outputs.
        '''
        self.root = self.recursive_build_tree(
            X, y, curr_depth=0, name='0')
            
    def recursive_build_tree(self, X, y, curr_depth, name):
        """
        This function recursively builds the tree. 

        Input: X -> examples; y -> classes; curr_depth -> current depth of the tree; name -> name of the node.

        Output: node -> the node that we just built.
        """
        node = Node()
        node.name = name
        node.X = X
        cls, cnts = np.unique(y, return_counts = True)
        # stop split when attrs is empty, or classes are same, or reached the max_depth.
        stop_cond = [len(cnts) == 1, X.shape[1] == 0, curr_depth == self.max_depth]
        if any(stop_cond):
            stop_split = True
        # if not meet stop condition, find the best attribute and split point to split.
        else:
            best = self.split_data(X, y)
            # if there's no appropriate split, then stop split.
            if not best:
                stop_split = True
            else:
                stop_split = False
        # if split stops, then make the node a leaf
        if stop_split:
            node.node_type = "leaf"
            node.predicted_class = scipy.stats.mode(y, keepdims=True)[0][0]
        # if we can split it, then split it into two parts and recursively build the tree.
        else:
            node.node_type = "nonleaf"
            node.test_attribute = best["bst_attr"]
            node.test_value = best["bst_split"]
            left_child = self.recursive_build_tree(best["L"], best["L_y"], curr_depth+1, name+".0")
            right_child = self.recursive_build_tree(best["R"], best["R_y"], curr_depth+1, name+".1")
            node.children.append(left_child)
            node.children.append(right_child)
        return node
    
    def predict(self, testset):
        """
        This function takes in a testset and returns an array of predictions.
        Assuming the testset has the same number of columns as the training set.

        Input: testset -> a 2-d numpy array of examples.

        Output: predction -> a 1-d numpy array of predictions.
        """
        predction = np.empty(len(testset), dtype=np.int32)
        for i in range(len(testset)):
            predction[i] = self.recursive_predict(testset[i], self.root)
        return predction
    
    def recursive_predict(self, example, node):
        """
        This function takes in a single eample and recursively predicts the class of it.

        Input: example -> a 1-d numpy array of an example; node -> node being tested.

        Output: recursively test a child node or return the predicted class of the example.
        """
        if node.node_type == "leaf":
            return node.predicted_class
        elif example[node.test_attribute] < node.test_value:
            return self.recursive_predict(example, node.children[0])
        return self.recursive_predict(example, node.children[1])

    def print(self):
        self.recursive_print(self.root)
    
    def recursive_print(self, node):
        print(node)
        for u in node.children:
            self.recursive_print(u)
            
    def entropy(self, y):
        """
        Return the information entropy in 1-d array y.
        """
        _, counts = np.unique(y, return_counts = True)
        probs = counts/counts.sum()
        return -(np.log2(probs) * probs).sum()
    
    def split_data(self, X, y):
        """
        This func goes over all attrs and their values to find the attr and split pos that give the most info gain.
        
        Input: X -> examples; y -> classes
        
        Output: return a dictionary with the following key/value pairs:
                (bst_attr -> index of the best attribute in X;
                bst_split -> value of the best split position for the best attribute;
                L -> examples in X that go into the left child;
                R -> examples in X that go into the right child.
                L_y -> corresponidng classes of examples in X that go into the left child;
                R_y -> corresponidng classes of examples in X that go into the right child.)
        """
        num_exs, num_attr = X.shape
        # keep track of the best attribute and split position
        bst_attr, bst_split, lst_entropy = -1, -1, np.inf
        for i in range(num_attr):
            vals = np.unique(X[:,i])
            if vals.shape[0] == 1:
                continue
            for k in range(vals.shape[0]-1):
                split = (vals[k] + vals[k+1]) / 2
                mask = X[:, i] < split
                L, L_y = X[mask], y[mask]
                R, R_y = X[~mask], y[~mask]
                l_len, r_len, len = L.shape[0], R.shape[0], X.shape[0]
                cost = (l_len / len) * self.entropy(L_y) + (r_len / len) * self.entropy(R_y) # the weighted entropy
                if cost < lst_entropy:
                    lst_entropy = cost
                    bst_attr = i
                    bst_split = split
                    b_L, b_R, bL_y, bR_y = np.array(L), np.array(R), np.array(L_y), np.array(R_y)
        if bst_attr == -1: # this means somehow we can't find a good split
            return None
        return {'bst_attr': bst_attr, 'bst_split': bst_split, 'L': b_L, 'R': b_R, 'L_y': bL_y, 'R_y': bR_y}
    

def validation_curve(filename, l, r, step):
    """
    This function plots the validation curve for the max_depth parameter.

    Input: filename -> the name of the file to read the data from;
            l -> the left bound of the range of max_depth to test;
            r -> the right bound of the range of max_depth to test;  
            step -> the step size of the range of max_depth to test.
    
    Output: a plot of the validation curve.
    """
    start = time.perf_counter()
    # Read the data into a pandas dataframe
    df = pd.read_csv(filename, header=None, na_values="?")

    # Replace each missing value with the mode
    a = df.shape[1]
    for i in range(a):
        if df[i].isnull().sum() > 0:
            df.iloc[:,i].fillna(df[i].mode()[0], inplace=True)

    # Split the data into 3 sets of roughly equal size
    set1, set2, set3 = np.split(df.sample(frac=1), [int(.33*len(df)), int(.66*len(df))])
    training1 = pd.concat([set1, set2])
    training2 = pd.concat([set2, set3])
    training3 = pd.concat([set1, set3])

    train_X_1, train_y_1 = training1.iloc[:, :-1].values, training1.iloc[:, -1].values
    test_X_1, test_y_1 = set3.iloc[:, :-1].values, set3.iloc[:, -1].values
    train_X_2, train_y_2 = training2.iloc[:, :-1].values, training2.iloc[:, -1].values
    test_X_2, test_y_2 = set1.iloc[:, :-1].values, set1.iloc[:, -1].values
    train_X_3, train_y_3 = training3.iloc[:, :-1].values, training3.iloc[:, -1].values
    test_X_3, test_y_3 = set2.iloc[:, :-1].values, set2.iloc[:, -1].values

    def get_accuracy(train_X, train_y, test_X, test_y, depth):
        """
        This function returns the accuracy of the decision tree with max_depth = depth.
        """
        tree = DecisionTree(max_depth=depth)
        tree.fit(train_X, train_y)

        train_pred = tree.predict(train_X)
        train_accuracy = np.mean(train_pred == train_y)

        test_pred = tree.predict(test_X)
        test_accuracy = np.mean(test_pred == test_y)

        return train_accuracy, test_accuracy
    
    train_accuracy = []
    test_accuracy = []
    y = []
    for i in range(l, r+1, step):
        y.append(i)
        train_accu1, test_accu1 = get_accuracy(train_X_1, train_y_1, test_X_1, test_y_1, i)
        train_accu2, test_accu2 = get_accuracy(train_X_2, train_y_2, test_X_2, test_y_2, i)
        train_accu3, test_accu3 = get_accuracy(train_X_3, train_y_3, test_X_3, test_y_3, i)
        train_accuracy.append((train_accu1 + train_accu2 + train_accu3) / 3)
        test_accuracy.append((test_accu1 + test_accu2 + test_accu3) / 3)

    elspsed = time.perf_counter() - start
    print(f"Finished in: {elspsed:0.4f} seconds")

    plt.plot(y, train_accuracy, label='train', marker='o')
    plt.plot(y, test_accuracy, label='test', marker='o')
    plt.xlabel('Max_depth')
    plt.ylabel('Average Accuracy')
    plt.legend()
    plt.title('Validation Curve on Arrhythmia Dataset')
    plt.savefig('validation_curve.png')
    plt.show()
    return 


if __name__ == "__main__":
    validation_curve("./arrhythmia.csv", 2, 16, 2)
