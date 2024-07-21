from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge

import numpy as np
from LSR_Tensor_2D_v1 import LSR_tensor_dot
from lsr_bcd_regression import lsr_bcd_regression
from optimization import inner_product, R2, objective_function_vectorized

def KFoldCV(X_train: np.ndarray, Y_train: np.ndarray, alphas, k_folds, hypers, B_tensored = None, intercept = False):
  
  #now alpha contain 2 rows for lambda 1 and lambda 2
  
  alpha_one = alphas[0]
  alpha_two = alphas[1]
  
  kfold = KFold(n_splits=k_folds, shuffle=True)

  #Matrix storing Validation Results
  if B_tensored is not None:
    validation_normalized_estimation_error = np.ones(shape = (k_folds, len(alpha_one),len(alpha_two))) * np.inf
  validation_nmse_losses = np.ones(shape = (k_folds, len(alpha_one),len(alpha_two)))* np.inf
  validation_correlations = np.ones(shape = (k_folds, len(alpha_one),len(alpha_two)))* np.inf
  validation_R2_scores = np.ones(shape = (k_folds, len(alpha_one),len(alpha_two)))* np.inf

  #Define LSR Tensor Hyperparameters
  ranks = hypers['ranks']
  separation_rank = hypers['separation_rank']
  LSR_tensor_dot_shape = tuple(X_train.shape)[1:]
  need_intercept = intercept

  #Initialize LSR Tensors for each alpha iteration
  lsr_tensors = [ [ [LSR_tensor_dot(shape = LSR_tensor_dot_shape, ranks = ranks, separation_rank = separation_rank, intercept = need_intercept) for fold_idx in range(k_folds)] for idx_1 in range(len(alpha_one))] for idx_2 in range(len(alpha_two))  ]

  #Store objective function values for each fold/alpha

  #objective_function_information = np.ones(shape = (k_folds, len(alphas), hypers['max_iter'], separation_rank, len(ranks) + 1))
  #gradient_information = np.ones(shape = (k_folds, len(alphas), hypers['max_iter'], separation_rank, len(ranks) + 1))
  
  #Go thru each fold
  #to handle errors
  for fold, (train_ids, validation_ids) in enumerate(kfold.split(X_train)):
      X_train_updated, Y_train_updated = X_train[train_ids], Y_train[train_ids]
      X_validation, Y_validation = X_train[validation_ids], Y_train[validation_ids]

      for index1, alpha1 in enumerate(alpha_one):
          for index2,alpha2 in enumerate(alpha_two):
           try:
              
              hypers['weight_decay'] = [alpha1,alpha2]

              lsr_ten, objective_function_values,gradient_values,iterate_level_values,factor_core_iteration = lsr_bcd_regression(lsr_tensors[index2][index1][fold], X_train_updated, Y_train_updated, hypers, intercept = need_intercept)
              expanded_lsr = lsr_ten.expand_to_tensor()
              expanded_lsr = np.reshape(expanded_lsr, X_validation[0].shape, order='F')
              Y_validation_predicted = inner_product(np.transpose(X_validation, (0, 2, 1)), expanded_lsr.flatten(order ='F')) + lsr_ten.b

              if B_tensored is not None:  
                normalized_estimation_error = ((np.linalg.norm(expanded_lsr - B_tensored)) ** 2) /  ((np.linalg.norm(B_tensored)) ** 2)

              #print(f"Y_validation_predicted: {Y_validation_predicted.flatten()}, Y_validation: {Y_validation.flatten()}")
              validation_nmse_loss = np.sum(np.square((Y_validation_predicted.flatten() - Y_validation.flatten()))) / np.sum(np.square(Y_validation.flatten()))
              correlation = np.corrcoef(Y_validation_predicted.flatten(), Y_validation.flatten())[0, 1]
              R2_value = R2(Y_validation.flatten(), Y_validation_predicted.flatten())

              if B_tensored is not None:
                validation_normalized_estimation_error[fold, index1,index2] = normalized_estimation_error   
              validation_nmse_losses[fold, index1,index2] = validation_nmse_loss
              validation_correlations[fold, index1,index2] = correlation
              validation_R2_scores[fold, index1,index2] = R2_value

              #Store Objective Function Information
              #objective_function_information[fold, index1] = objective_function_values
              #gradient_information[fold, index1] = gradient_values 

              if B_tensored is not None:
                print(f"Fold = {fold}, Alpha = {alpha1,alpha2}, Normalized Estimation Error: {normalized_estimation_error}, NMSE: {validation_nmse_loss}, Correlation: {correlation}, R^2 Score: {R2_value}")
              else:
                print(f"Fold = {fold}, Alpha = {alpha1,alpha2}, NMSE: {validation_nmse_loss}, Correlation: {correlation}, R^2 Score: {R2_value}")

           except Exception as e:
              #Handle the error and continue with the next lambda value
              print(f"Fold:{fold} = {fold} Lambda {alpha1}: Error occurred during cross-validation: {e}")
              continue
      
  #Average out validation results
  if B_tensored is not None:
      average_normalized_estimation_error = np.mean(validation_normalized_estimation_error, axis = 0)
  average_validation_nmse_losses = np.mean(validation_nmse_losses, axis = 0)
  average_validation_correlations = np.mean(validation_correlations, axis = 0)
  average_validation_R2_scores = np.mean(validation_R2_scores, axis = 0)

  #Get alpha value that performs the best
  #flattened_avg_validation_nmse_losses = average_validation_nmse_losses.flatten()
  min_value_coordinates = np.unravel_index(np.argmin(average_validation_nmse_losses), average_validation_nmse_losses.shape)
  lambda1 = [alpha_one[min_value_coordinates[0]],alpha_two[min_value_coordinates[1]]]
  
  if B_tensored is not None:
    return lambda1, validation_normalized_estimation_error, validation_nmse_losses, validation_correlations, validation_R2_scores
  else: 
    validation_normalized_estimation_error = np.inf
    normalized_estimation_error = np.inf
    return lambda1, validation_normalized_estimation_error, validation_nmse_losses, validation_correlations, validation_R2_scores





