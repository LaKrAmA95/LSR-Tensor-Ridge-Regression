### CODE FOR DATA GENERATION for SGD Experiments
import numpy as np
from sklearn.model_selection import train_test_split

#Generate Data for SGD Experiments
def generate_data(n_train, n_test, d, intercept = False):
    X = np.random.normal(size = (n_train + n_test, d))
    W = np.random.normal(size = (d, 1))
    b = np.random.normal() if intercept else 0
    Y = X @ W + b
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.33, random_state = 42)
    return X_train, X_test, Y_train, Y_test, W, b
