import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier

def generate_classes(signal_magnitude, num_samples, num_predictors):
    scale  = 1.0
    noise  = 0.0
    signal = 1.0

    X = np.zeros((num_samples, num_predictors))
    Y = np.zeros((num_samples, num_predictors))

    for row in range(num_samples):
        X[row] = np.random.normal(loc=noise, scale=scale, size=num_predictors)
        Y[row] = np.random.normal(loc=noise, scale=scale, size=num_predictors)

    for row in range(num_samples):
        Y[row, :signal_magnitude] = np.random.normal(loc=signal, scale=scale, size=signal_magnitude)

    return X, Y

def create_samples_and_labels(num_samples, X, Y):
    x_labels = np.array([0] * num_samples)
    y_lables = np.array([1] * num_samples)
    samples  = np.concatenate((X, Y))
    labels   = np.concatenate((x_labels, y_lables))

    return samples, labels

def test_classifier(figure_num, signal_magnitude, model, num_train_samples, model_name):
    num_test_samples    = 200
    loss_list           = []
    num_predictors_list = range(10, 10000, 100)

    for num_predictors in num_predictors_list:
        X_train, Y_train = generate_classes(signal_magnitude, num_train_samples, num_predictors)
        X_test, Y_test   = generate_classes(signal_magnitude, num_test_samples, num_predictors)

        train_samples, train_labels = create_samples_and_labels(num_train_samples, X_train, Y_train)
        test_samples, test_labels   = create_samples_and_labels(num_test_samples, X_test, Y_test)

        model.fit(train_samples, train_labels)

        predictions = model.predict(test_samples)
        loss        = log_loss(test_labels, predictions)

        loss_list.append(loss)

        print(f"Signal Magnitude: {signal_magnitude}, Number of Predictors: {num_predictors}")

    plt.plot(num_predictors_list, loss_list, "+", label=model_name)

def main():
    svc_model = svm.SVC(kernel='linear', degree=2)
    dl_model  = MLPClassifier(solver='adam', alpha=1e-3, hidden_layer_sizes=(32, 8), random_state=1)

    figure_num = 0

    for _, signal_magnitude in enumerate([2]):
        plt.figure(figure_num)

        test_classifier(figure_num, signal_magnitude, svc_model, num_train_samples=1000, model_name="SVC")
        test_classifier(figure_num, signal_magnitude, dl_model, num_train_samples=10000, model_name="Deep Learning")

        plt.xlabel("number of predictors")
        plt.ylabel("error")
        plt.legend(["SVC", "Deep Learning"], loc='upper left')
        plt.savefig(f"results/signal magnitude = {signal_magnitude}.png")

        figure_num += 1

if __name__ == "__main__":
    main()