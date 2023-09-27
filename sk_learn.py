import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

def plot_err_Dbig():
    error_info = np.loadtxt(f'Homework 2 data/Dbig_predictions_error_sklearn.txt', delimiter=' ')
    plt.plot(error_info[:, 0], error_info[:, 2])
    plt.xlabel('Sample Size, n')
    plt.ylabel('Error, err(n)')
    plt.title('Learning Curve, Dbig Dataset (Sklearn)')
    plt.savefig('learning_curve_Dbig_sklearn')
    plt.show()

def main():
    """
    Read various Dbig_size training sets, Train the Sklearn Decision Tree
    Predict labels for Dbig_test, calculate error and plot learning cure
    """
    sample_sizes = [32, 128, 512, 2048, 8192]

    testing_set = pd.read_csv(f'Homework 2 data/Dbig_test.txt', sep=" ", header=None)
    testing_features, testing_labels = testing_set.iloc[:, 0:2], np.array(testing_set.iloc[:, -1])

    nodes_error_map = []

    for size in sample_sizes:
        training_set = pd.read_csv(f'Homework 2 data/Dbig_{size}.txt', sep=" ", header=None)
        training_set.columns = ['x1', 'x2', 'y']
        training_features, training_labels = training_set.iloc[:, 0:2], training_set.iloc[:, -1]

        decision_tree = DecisionTreeClassifier()
        decision_tree.fit(training_features, training_labels)

        testing_predictions = decision_tree.predict(np.array(testing_features))
        np.savetxt(f'Homework 2 data/Dbig_{size}_predictions_sklearn.txt', testing_predictions, delimiter=' ', fmt='%1.6f')

        test_error = np.square(np.subtract(testing_labels, testing_predictions)).mean()
        nodes_error_map.append([size, decision_tree.tree_.node_count, test_error])

    np.savetxt(f'Homework 2 data/Dbig_predictions_error_sklearn.txt', np.array(nodes_error_map), delimiter=' ', fmt='%1.6f')
    plot_err_Dbig()

if __name__ == '__main__':
    main()