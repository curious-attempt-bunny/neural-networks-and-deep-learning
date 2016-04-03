import numpy as np
import random

def load_csv():
    data_set = np.genfromtxt('../data/training.csv', delimiter=',')
    random.seed(10)
    random.shuffle(data_set)
    return data_set

def load_data(data):
    tr_d = data[0:0.8*len(data)]
    va_d = data[0.8*len(data):0.9*len(data)]
    te_d = data[0.9*len(data):len(data)]
    training_inputs = [np.reshape(x[0:320], (320, 1)) for x in tr_d]
    training_results = [vectorized_result(y[320:321]) for y in tr_d]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x[0:320], (320, 1)) for x in va_d]
    validation_results = [y[320:321] for y in va_d]
    validation_data = zip(validation_inputs, validation_results)
    test_inputs = [np.reshape(x[0:320], (320, 1)) for x in te_d]
    test_results = [y[320:321] for y in te_d]
    test_data = zip(test_inputs, test_results)
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    e = np.zeros((2, 1))
    e[int(j)] = 1.0
    return e
