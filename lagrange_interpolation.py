import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import lagrange

def add_noise(samples, standard_deviation):
    gaussian_noise = np.random.normal(0, standard_deviation, size = samples.size)
    samples_with_noise = samples + gaussian_noise
    return samples_with_noise

def error(labels, predictions):
    return np.log(np.square(np.subtract(labels, predictions)).mean())

def main():
    train_samples = np.random.uniform(-2.0, 2.0, 100)
    train_labels = np.sin(train_samples)

    test_samples = np.random.uniform(-2.0, 2.0, 100)
    test_labels = np.sin(test_samples)

    f = lagrange(train_samples, train_labels)
    
    train_predictions = f(train_samples)
    test_predictions = f(test_samples)

    train_error = error(train_labels, train_predictions)
    test_error = error(test_labels, test_predictions)

    np.savetxt(f'Homework 2 data/lagrange_errors.txt', [["Training Error", train_error], ["Testing Error", test_error]], delimiter=' ', fmt='%s')

    train_errors_noisy_samples = []
    test_errors_noisy_samples = []
    for std in np.arange(0, 5.5, 0.5):
        train_noise = add_noise(train_samples, std)
        train_labels = np.sin(train_noise)

        f = lagrange(train_noise, train_labels)

        train_noise_predictions = f(train_noise)
        test_predictions = f(test_samples)

        train_error = error(train_labels, train_noise_predictions)
        test_error = error(test_labels, test_predictions)

        train_errors_noisy_samples.append([std, train_error])
        test_errors_noisy_samples.append([std, test_error])
        
    np.savetxt(f'Homework 2 data/lagrange_train_errors_noise.txt', train_errors_noisy_samples, delimiter=' ', fmt='%s')
    np.savetxt(f'Homework 2 data/lagrange_test_errors_noise.txt', test_errors_noisy_samples, delimiter=' ', fmt='%s')

    standard_deviations = np.array(train_errors_noisy_samples)[:,0]
    log_error_values_train = np.array(train_errors_noisy_samples)[:,1]

    plt.plot(standard_deviations, log_error_values_train, label = 'Training Error')

    log_error_values_test = np.array(test_errors_noisy_samples)[:,1]

    plt.plot(standard_deviations, log_error_values_test, label = 'Testing Error')
    plt.xlabel("Standard Deviation for White Noise")
    plt.ylabel("Log Error")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()