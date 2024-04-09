### CODE FOR DATA GENERATION
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime

#n_train: number of training data samples
#n_test: number of test data samples
#dimension of each sample
def generate_data(n_train, n_test, d, intercept = False):
    X = np.random.normal(size = (n_train + n_test, d))
    W = np.random.normal(size = (d, 1))
    b = np.random.normal() if intercept else 0
    Y = X @ W + b
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    
    # Get the current date and time
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    
    #Store files in .npy files
    np.save(f"../Data/X_train_time={formatted_time}_train={n_train}_test={n_test}_d={d}_intercept={intercept}.npy", X_train)
    np.save(f"../Data/X_test_time={formatted_time}_train={n_train}_test={n_test}_d={d}_intercept={intercept}.npy", X_test)
    np.save(f"../Data/Y_train_time={formatted_time}_train={n_train}_test={n_test}_d={d}_intercept={intercept}.npy", Y_train)
    np.save(f"../Data/Y_test_time={formatted_time}_train={n_train}_test={n_test}_d={d}_intercept={intercept}.npy", Y_test)
    
    return X_train, X_test, Y_train, Y_test, W
