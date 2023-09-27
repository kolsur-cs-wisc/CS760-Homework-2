import argparse
import graphviz
from math import log2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class TreeNode():
    def __init__(self, feature, threshold, id, gain_ratio=0):
        self.id = id
        self.left = None
        self.right = None
        self.isLeaf = False
        self.feature = feature
        self.threshold = threshold
        self.gain_ratio = gain_ratio
        self.label = None

class DecisionTree():
    def __init__(self, data, features, labels):
        self.root = None
        self.data = data
        self.features = features
        self.labels = labels
        self.count = 1

    def __call__(self):
        self.root = self.make_tree(self.features, self.labels)
        # print(self.print_tree(self.root, 0))
        # self.show_decision_tree(self.root)
        return self.root

    def make_tree(self, features, labels):
        y1Count = np.count_nonzero(labels)
        y0Count = len(labels) - y1Count

        feature, threshold, max_gain_ratio = self.best_feature_split(features, labels)

        if max_gain_ratio == 0:
            leaf = TreeNode(None, None, id=self.count)
            self.count += 1
            leaf.isLeaf = True
            if y0Count > y1Count:
                leaf.label = 0
            else:
                leaf.label = 1
            return leaf
        
        else:
            node = TreeNode(feature, threshold, self.count, max_gain_ratio)
            self.count += 1
            data = pd.concat([features, labels], axis = 1)
            split1 = data[data[feature] >= threshold]
            split2 = data[data[feature] < threshold]

            node.left = self.make_tree(split1.iloc[:, 0:2], split1.iloc[:, -1])
            node.right = self.make_tree(split2.iloc[:, 0:2], split2.iloc[:, -1])

            return node

        
    def best_feature_split(self, features, labels):
        feature_to_split, best_threshold, max_gain_ratio = "", -1, -1

        for feature in features.columns:
            feature_values = list(set(np.array(features[feature], dtype=float)))
            threshold, curr_gain_ratio = self.get_feature_threshold(features, labels, feature, feature_values)
            if max_gain_ratio < curr_gain_ratio:
                max_gain_ratio = curr_gain_ratio
                best_threshold = threshold
                feature_to_split = feature

        return feature_to_split, best_threshold, max_gain_ratio

    
    def get_feature_threshold(self, features, labels, feature, feature_values):
        max_gain_ratio, best_threshold = -1, -1

        for curr_threshold in feature_values:
            curr_gain_ratio, curr_information_gain = self.info_gain_ratio(features, labels, feature, curr_threshold)
            # print(f'({feature} >= {curr_threshold}) --------- {curr_gain_ratio}, {curr_information_gain}')
            if max_gain_ratio < curr_gain_ratio:
                max_gain_ratio = curr_gain_ratio
                best_threshold = curr_threshold

        return best_threshold, max_gain_ratio
                

    def info_gain_ratio(self, features, labels, feature, threshold):
        data = pd.concat([features, labels], axis = 1)
        split1 = data[data[feature] >= threshold]['y']
        split2 = data[data[feature] < threshold]['y']

        ps1 = len(split1)/len(labels)
        ps2 = len(split2)/len(labels)

        if ps1 == 1 or ps2 == 1:
            return 0.0, 0.0

        information_gain = self.empirical_entropy(labels) - self.conditional_entropy(labels, split1, split2)

        split_information = - (ps1 * log2(ps1) + ps2 * log2(ps2))
        
        return information_gain/split_information, information_gain

    def empirical_entropy(self, labels):
        h_y = -1
        total = len(labels)

        if total == 0:
            return 0
        
        y1Count = np.count_nonzero(labels)
        y0Count = total - y1Count

        py1 = y1Count/total
        py0 = y0Count/total

        if py0 == 1 or py1 == 1:
            h_y = 0
        else:
            h_y = - (py0 * log2(py0) + py1 * log2(py1))
        return h_y

    def conditional_entropy(self, labels, split1, split2):
        total = len(labels)
        h_yx = (len(split1)/total) * self.empirical_entropy(split1) + (len(split2)/total) * self.empirical_entropy(split2)
        return h_yx
    
    def tree_prediction_util(self, root, features, feature_map):
        if root.isLeaf == True:
            return root.label
        if features[feature_map[root.feature]] >= root.threshold:
            return self.tree_prediction_util(root.left, features, feature_map)
        return self.tree_prediction_util(root.right, features, feature_map)
    
    def tree_prediction(self, features):
        feature_map = {'x1':0, 'x2':1}
        predicted_labels = np.array([self.tree_prediction_util(self.root, inputs, feature_map) for inputs in features])
        return predicted_labels
    
    def print_tree(self, root, level):
        if root == None:
            return ""
        
        view = "\t"*level + f'Threshold:{root.threshold}, Feature:{root.feature}, Gain Ratio:{root.gain_ratio:.4f}, Leaf:{root.isLeaf}, Label:{root.label}, Level:{level}' + "\n"
        if root.left:
            view += self.print_tree(root.left, level+1)
        if root.right:
            view += self.print_tree(root.right, level+1)
        
        return view
    
    def show_decision_tree(self, root, file):
        tree = graphviz.Digraph()
        if root.feature:
            root_key = str(f'{root.feature} >= {root.threshold} ({root.id})')
        else:
            root_key = str(f'Y = {root.label} ({root.id})')
        tree.node(root_key)

        def add_edges(node):
            if node.feature:
                node_key = str(f'{node.feature} >= {node.threshold} ({node.id})')
            else:
                node_key = str(f'Y = {node.label} ({node.id})')

            if node.left:
                if node.left.feature:
                    left_key = str(f'{node.left.feature} >= {node.left.threshold} ({node.left.id})')
                else:
                    left_key = str(f'Y = {node.left.label} ({node.left.id})')

                tree.node(left_key)
                tree.edge(str(node_key), str(left_key))
                add_edges(node.left)
            if node.right:
                if node.right.feature:
                    right_key = str(f'{node.right.feature} >= {node.right.threshold} ({node.right.id})')
                else:
                    right_key = str(f'Y = {node.right.label} ({node.right.id})')
                tree.node(right_key)
                tree.edge(str(node_key), str(right_key))
                add_edges(node.right)

        add_edges(root)
        tree.render(f'decision_tree_{file}', view=True, format='png')

    def decision_region(self, dataset_name):
        range = np.linspace(0.0, 0.9, 600)
        grid = np.array(np.meshgrid(range, range)).T.reshape(-1, 2)

        data = pd.DataFrame(grid)
        data.columns = ['x1', 'x2']
        data['y'] = self.tree_prediction(grid)
        feature_x1, feature_x2, label = data.columns[0], data.columns[1], data.columns[-1]
        # data = data.sort_values(by=label)

        for prediction in [0, 1]:
            df_set = data[data[label]==prediction]
            set_x = df_set[feature_x1]
            set_y = df_set[feature_x2]
            plt.scatter(set_x,set_y,label=f'y = {prediction}',marker='s')

        plt.xlabel('Feature x1')
        plt.ylabel('Feature x2')
        plt.title(f"Decision Boundary for {dataset_name}")
        plt.legend()
        plt.savefig(f'decision_boundary_{dataset_name}.png')
        plt.show()

def scatter_plot(data):
    data_y0 = data[data['y'] == 0]
    plt.scatter(data_y0.iloc[:, 0], data_y0.iloc[:, 1], label = "y = 0")

    data_y1 = data[data['y'] == 1]
    plt.scatter(data_y1.iloc[:, 0], data_y1.iloc[:, 1], label = "y = 1")

    # boundary_x1 = np.arange(0, 1.05, 0.05)
    # boundary_x2 = np.repeat([0.201829], len(boundary_x1))
    # plt.plot(boundary_x1, boundary_x2, linestyle = 'dashed', color = 'red', linewidth=2)

    plt.xlabel('Feature x1')
    plt.ylabel('Feature x2')
    plt.title('Data Set D2')
    plt.legend()
    plt.savefig('scatter_plot_D2')
    plt.show()

def plot_err_Dbig():
    error_info = np.loadtxt(f'Homework 2 data/Dbig_predictions_error.txt', delimiter=' ')
    plt.plot(error_info[:, 0], error_info[:, 2])
    plt.xlabel('Sample Size, n')
    plt.ylabel('Error, err(n)')
    plt.title('Learning Curve, Dbig Dataset')
    plt.savefig('learning_curve_Dbig')
    plt.show()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="Homework 2 data/D2.txt")
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    data = pd.read_csv(args.file, sep=" ", header=None)

    """
    Read D_big.txt, sample 8192 rows, split into training sets of various sizes and save them to text file. 
    Data not part of sample is part of testing set and is saved to Dbig_test file.

    Run loop over each training set, build it's decision tree and calculate error in the testing set. 
    """
    if "Dbig" in args.file:
        sample_sizes = [32, 128, 512, 2048, 8192]

        if "Dbig.txt" in args.file:
            d_8192 = data.sample(8192)
            for size in sample_sizes:
                d_size = d_8192.iloc[0:size]
                np.savetxt(f'Homework 2 data/Dbig_{size}.txt', d_size.values, delimiter=' ', fmt='%1.6f')
            d_test = data[~data.isin(d_8192)].dropna()
            np.savetxt(f'Homework 2 data/Dbig_test.txt', d_test.values, delimiter=' ', fmt='%1.6f')

        testing_set = pd.read_csv(f'Homework 2 data/Dbig_test.txt', sep=" ", header=None)
        testing_features, testing_labels = testing_set.iloc[:, 0:2], np.array(testing_set.iloc[:, -1])

        nodes_error_map = []

        for size in sample_sizes:
            training_set = pd.read_csv(f'Homework 2 data/Dbig_{size}.txt', sep=" ", header=None)
            training_set.columns = ['x1', 'x2', 'y']
            training_features, training_labels = training_set.iloc[:, 0:2], training_set.iloc[:, -1]

            decision_tree = DecisionTree(training_set, training_features, training_labels)
            decision_tree_root = decision_tree()
            print(f'Node Count for Sample Size {size}: {decision_tree.count - 1}')

            testing_predictions = decision_tree.tree_prediction(np.array(testing_features))
            np.savetxt(f'Homework 2 data/Dbig_{size}_predictions.txt', testing_predictions, delimiter=' ', fmt='%1.6f')

            test_error = np.square(np.subtract(testing_labels, testing_predictions)).mean()
            nodes_error_map.append([size, decision_tree.count - 1, test_error])

            decision_tree.decision_region(f'D{size}')

        np.savetxt(f'Homework 2 data/Dbig_predictions_error.txt', np.array(nodes_error_map), delimiter=' ', fmt='%1.6f')
        plot_err_Dbig()
        return
    
    data.columns = ['x1', 'x2', 'y']
    features = data.iloc[:, 0:2] 
    labels = data.iloc[:, -1]

    scatter_plot(data)
    
    decision_tree = DecisionTree(data, features, labels)
    decision_tree_root = decision_tree()
    print(f'Node Count: {decision_tree.count}')
    file_name = args.file.split('/')[1].split('.')[0]
    decision_tree.show_decision_tree(decision_tree_root, file_name)
    decision_tree.decision_region(file_name)

if __name__ == '__main__':
    main()