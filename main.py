import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm, datasets, tree
from sklearn.metrics import log_loss, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import time
import random

SIGNAL_MAGNITUDE = 2

def generate_trivial_classes(num_samples, num_predictors):
    # generates a randomly generated dataset consisting of two seperate data classes. The first is a collection
    # of vectors in which every element of every vector is a gaussian random variable centered at 0. The second
    # class is a collection of vectors such that the first SIGNAL_MAGNITUDE elements are gaussian randomly
    # generated values centered at signal (set at 1.0), and the rest of the elements are gaussian random variables
    # centered at 0.0. 

    scale  = 1.0
    noise  = 0.0
    signal = 1.0

    # randomly generating the labels
    labels = np.random.randn(num_samples)
    labels = [1 if label > .5 else 0 for label in labels]

    data = np.zeros((num_samples, num_predictors))

    for row in range(num_samples):
        data[row] = np.random.normal(loc=noise, scale=scale, size=num_predictors)

        # if label == 1, the first SIGNAL_MAGNITUDE will be the "signal"
        if labels[row] == 1:
            data[row, :SIGNAL_MAGNITUDE] = np.random.normal(loc=signal, scale=scale, size=SIGNAL_MAGNITUDE)

    return data, labels

def generate_nontrivial_classes(num_samples, num_predictors):
    # Generates a dataset consisting of the '0' and '1' digits from the MNIST dataset. Adds padding to these 
    # digits such that the total number of elements in the image matches num_predictors. Randomly generates
    # this padding so that the image is randomly placed somewhere in the overall image. Adds noise to each 
    # element in the image. 

    digits = datasets.load_digits()
    images = []

    for i, target in enumerate(digits.target):
        if target in [0, 1]:
            images.append((target, digits.images[i]))

    random.shuffle(images)
    data, labels = [], []

    padding = max(0, int(np.sqrt(num_predictors)) - 8)

    for i in range(num_samples):
        index         = i % len(images)
        target, image = images[index]

        padding_left   = random.randint(0, padding)
        padding_right  = padding - padding_left
        padding_top    = random.randint(0, padding)
        padding_bottom = padding - padding_top

        digit  = (image / np.amax(image))

        digit  = np.pad(
            digit, 
            [
                (padding_left, padding_right), 
                (padding_top, padding_bottom)
            ], 
            mode='constant', 
            constant_values=0
        )

        digit += (np.random.random(digit.shape) / 2) - .5

        # plt.imshow(digit, cmap='hot', interpolation='nearest')
        # plt.savefig(f"results/digit_{num_predictors}.png")
        # print(target)
        # time.sleep(1)

        data.append(digit.flatten())
        labels.append(target)

    return data, labels

def test_classifier(model, model_name, generate_classes_func):
    num_train_samples = 1000
    num_test_samples  = 100

    loss_list           = []
    num_predictors_list = range(SIGNAL_MAGNITUDE, 1000, 10)

    for num_predictors in num_predictors_list:
        train_x, train_y = generate_classes_func(num_train_samples, num_predictors)
        test_x, test_y   = generate_classes_func(num_test_samples, num_predictors)

        model.fit(train_x, train_y)

        predictions = model.predict(test_x)
        loss        = accuracy_score(test_y, predictions)
        loss_list.append(loss)

        print(f"Signal Magnitude: {SIGNAL_MAGNITUDE}, Number of Predictors: {num_predictors}, accuracy: {loss}")

    return num_predictors_list, loss_list

def get_running_average(loss_list):
    running_avg_list   = []
    running_avg_lenght = 10

    for i in range(len(loss_list)):
        temp_list   = loss_list[max(0, i - running_avg_lenght) : i + 1]
        running_avg = sum(temp_list) / len(temp_list)
        running_avg_list.append(running_avg)

    return running_avg_list

if __name__ == "__main__":
    svm_model = svm.SVC(kernel='linear', degree=2)
    dl_model  = MLPClassifier(solver='adam', alpha=1e-3, hidden_layer_sizes=(32, 16), random_state=1)
    dtc_model = tree.DecisionTreeClassifier()

    svm_x, svm_loss_list = test_classifier(svm_model, "SVC", generate_nontrivial_classes)
    dl_x, dl_loss_list   = test_classifier(dl_model, "Deep Learning", generate_nontrivial_classes)
    dtc_x, dtc_loss_list = test_classifier(dtc_model, "Dicision Tree", generate_nontrivial_classes)

    plt.plot(svm_x, svm_loss_list, "+", label="SVM", color="lightcoral")
    plt.plot(dl_x, dl_loss_list, "+", label="DL", color="lightsteelblue")
    plt.plot(dtc_x, dtc_loss_list, "+", label="DTC", color="palegreen")

    svm_running_avg_list = get_running_average(svm_loss_list)
    dl_running_avg_list  = get_running_average(dl_loss_list)
    dtc_running_avg_list = get_running_average(dtc_loss_list)

    plt.plot(svm_x, svm_running_avg_list, label="SVM - avg.", color="red")
    plt.plot(dl_x, dl_running_avg_list, label="DL - avg.", color="blue")
    plt.plot(dtc_x, dtc_running_avg_list, label="DTC - avg.", color="green")

    plt.plot([SIGNAL_MAGNITUDE], [0])
    plt.xlabel("number of predictors")
    plt.ylabel("percent correct")
    plt.title("Success Rate vs Total Number of Predictors")
    plt.legend(["SVM", "Deep Learning", "Decision Tree", "SVM - average", "DL - average", "DT - average"], loc='lower left')
    plt.savefig(f"results/signal magnitude = {SIGNAL_MAGNITUDE}.png")