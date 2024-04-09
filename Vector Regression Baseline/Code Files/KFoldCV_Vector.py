from torchmetrics import R2Score
import numpy as np
from sklearn.model_selection import KFold
from optimization_Vector import optimization_subroutine
import torch

#Run the KFold Cross Validation with X_train, Y_train, and the ground truth Weights
#X_train: Shape n x d where n is the number of samples and d is the number of features
#Y_train: Shape n x 1 where n is the number of samples
#W: Shape d x 1 where d is the number of features
def KFoldCV(X_train: np.ndarray, Y_train: np.ndarray, W: np.ndarray, lambdas, k_folds, intercept = False):
    #Set up KFold Object
    kfold = KFold(n_splits = k_folds, shuffle = True)

    #Set up Matrices to Store Validation Results
    validation_normalized_estimation_errors = np.zeros(shape = (k_folds, len(lambdas)))
    validation_nmse_losses = np.zeros(shape = (k_folds, len(lambdas)))
    validation_correlations = np.zeros(shape = (k_folds, len(lambdas)))
    validation_R2_scores = np.zeros(shape = (k_folds, len(lambdas)))
    objective_function_values = np.zeros(shape = (k_folds, len(lambdas)))

    #Run through each Fold of KFold CV
    for fold, (train_ids, validation_ids) in enumerate(kfold.split(X_train)):
        X_train_updated, Y_train_updated = X_train[train_ids], Y_train[train_ids] #Keep n - 1 Folds for Training
        X_validation, Y_validation = X_train[validation_ids], Y_train[validation_ids] #Hold one out for validation

        #Go through each lambda value to train a model on the training folds
        for index1, lambda1 in enumerate(lambdas):
            #Fit LRR on X_train_updated, Y_train_updated
            W_estimated, b = optimization_subroutine(X_train_updated, Y_train_updated, lambda1, intercept)

            #Using the Fitted Model, generate Y_validation_predicted
            Y_validation_predicted = X_validation @ W_estimated + b

            #Compute NEE, NMSE, Correlation, and R^2 Score
            validation_normalized_estimation_error = ((np.linalg.norm(W_estimated - W)) ** 2) /  ((np.linalg.norm(W)) ** 2)
            validation_nmse_loss = np.sum(np.square((Y_validation_predicted - Y_validation))) / np.sum(np.square(Y_validation))
            validation_correlation = np.corrcoef(Y_validation_predicted.flatten(), Y_validation.flatten())[0, 1]
            
            print(W_estimated.shape, Y_validation_predicted.shape, Y_validation.shape)
            r2 = R2Score()
            r2.update(torch.tensor(Y_validation_predicted), torch.tensor(Y_validation))
            validation_R2_score = r2.compute()

            #the intercept 
            print('Train Intercept:',b)

            #Compute objective function value
            validation_objective_function_value = (np.linalg.norm(Y_validation - (X_validation @ W_estimated + b)) ** 2) + (lambda1 * (np.linalg.norm(W_estimated) ** 2))

            #Store NEE, NMSE, Correlation, and R^2 Score in Matrices
            validation_normalized_estimation_errors[fold, index1] = validation_normalized_estimation_error
            validation_nmse_losses[fold, index1] = validation_nmse_loss
            validation_correlations[fold, index1] = validation_correlation
            validation_R2_scores[fold, index1] = validation_R2_score
            objective_function_values[fold, index1] = validation_objective_function_value

            #Print Results
            print(f"Fold = {fold}, lambda = {lambda1}, NEE: {validation_normalized_estimation_error}, NMSE: {validation_nmse_loss}, Correlation: {validation_correlation}, R^2 Score: {validation_R2_score}, Objective Function Value: {validation_objective_function_value}")

    #Average out validation results
    average_validation_normalized_estimation_errors = np.mean(validation_normalized_estimation_errors, axis = 0)
    average_validation_nmse_losses = np.mean(validation_nmse_losses, axis = 0)
    average_validation_correlations = np.mean(validation_correlations, axis = 0)
    average_validation_R2_scores = np.mean(validation_R2_scores, axis = 0)

    #Get lambda value that performs the best
    flattened_avg_validation_nmse_losses = average_validation_nmse_losses.flatten()
    best_lambda = lambdas[np.argmin(flattened_avg_validation_nmse_losses)]

    #Return best lambda as well as ALL Validation Results
    return best_lambda, validation_normalized_estimation_errors, validation_nmse_losses, validation_correlations, validation_R2_scores, objective_function_values