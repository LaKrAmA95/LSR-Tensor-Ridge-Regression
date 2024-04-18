from torchmetrics import R2Score
import numpy as np
from optimization_Vector import optimization_subroutine
import torch

#Given the Best Lambda found by KFoldCV, train a LRR model with that best lambda and generate test metrics!
#X_train: Shape n x d where n is the number of samples and d is the number of features
#Y_train: Shape n x 1 where n is the number of samples
#W: Shape d x 1 where d is the number of features
def TrainTest(X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray, Y_test: np.ndarray, W: np.ndarray, best_lambda, intercept = False):
  #Fit LRR on X_train_updated, Y_train_updated
  W_estimated, b = optimization_subroutine(X_train, Y_train, best_lambda, intercept)

  #Using the Fitted Model, generate Y_validation_predicted
  Y_test_predicted = X_test @ W_estimated + b

  #Compute NEE, NMSE, Correlation, and R^2 Score
  test_normalized_estimation_error = ((np.linalg.norm(W_estimated - W)) ** 2) /  ((np.linalg.norm(W)) ** 2)
  test_nmse_loss = np.sum(np.square((Y_test_predicted - Y_test))) / np.sum(np.square(Y_test))  
  test_correlation = np.corrcoef(Y_test_predicted.flatten(), Y_test.flatten())[0, 1]

  r2 = R2Score()
  r2.update(torch.tensor(Y_test_predicted), torch.tensor(Y_test))
  test_R2_score = r2.compute()

  #Print Test Results
  print(f"NEE: {test_normalized_estimation_error}, NMSE: {test_nmse_loss}, Correlation: {test_correlation}, R^2 Score: {test_R2_score}, Intercept: {b}")

  #Return Test Results
  return test_normalized_estimation_error, test_nmse_loss, test_correlation, test_R2_score, Y_test_predicted