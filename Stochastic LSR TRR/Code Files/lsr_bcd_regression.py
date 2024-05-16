from sklearn.linear_model import Ridge
import LSR_Tensor_2D_v1
import numpy as np
from optimization import objective_function_tensor
from sklearn.linear_model import SGDRegressor
from sgd_optimization import SGD

#lsr_ten: LSR Tensor
#training_data: X
#training_labels: Y
#hypers: hyperparameters
def lsr_bcd_regression(lsr_ten, training_data: np.ndarray, training_labels: np.ndarray, hypers: dict, intercept = False):
    #Get LSR Tensor Information and other hyperparameters
    shape, ranks, sep_rank, order = lsr_ten.shape, lsr_ten.ranks, lsr_ten.separation_rank, lsr_ten.order
    lambda1 = hypers["weight_decay"]
    max_iter = hypers["max_iter"]
    threshold = hypers["threshold"]
    lr        = hypers["learning_rate"]
    epochs    = hypers["epochs"]
    batch_size = hypers["batch_size"]
    b_intercept = intercept

    #Create models for each factor matrix and core matrix
    #factor_matrix_models = [[Ridge(alpha = lambda1, solver = 'svd', fit_intercept = intercept) for k in range(len(ranks))] for s in range(sep_rank)]
    #core_tensor_model = Ridge(alpha = lambda1, solver = 'svd', fit_intercept = intercept)

    #Store objective function values
    objective_function_values = np.ones(shape = (max_iter, sep_rank, len(ranks) + 1)) * np.inf

    #Normalized Estimation Error
    iterations_normalized_estimation_error = np.zeros(shape = (max_iter,))

    #Run at most max_iter iterations of Block Coordinate Descent
    for iteration in range(max_iter):
        #print('')
        #print('--------------------------------------------------------------BCD iteration',iteration,'--------------------------------------------------------------')
        factor_residuals = np.zeros(shape = (sep_rank, len(ranks)))
        core_residual = 0

        #Store updates to factor matrices and core tensor
        updated_factor_matrices = np.empty((sep_rank, len(ranks)), dtype=object)
        updated_core_tensor = None

        #Iterate over the Factor Matrices.
        for s in range(sep_rank):
            for k in range(len(ranks)):
                #Absorb Factor Matrices into X aside from (s, k) to get X_tilde
                #print('')
                #print('---------------------------------------------Sep',s,'Factor',k,'-------------------------------------------------')
                X, y = training_data, training_labels
                X_tilde, y_tilde = lsr_ten.bcd_factor_update_x_y(s, k, X, y) #y tilde should now be y-b-<Q,X>
                

                #Solve the sub-problem pertaining to the factor tensor
                hypers = {'lambda': lambda1, 'lr': lr, 'epochs': epochs, 'batch_size': batch_size, 'bias': b_intercept}
                weights, bias, loss_values, gap_to_optimality, nmse_values, corr_values, R2_values = SGD(X_tilde, y_tilde.reshape(-1,1), cost_function_code = 1, hypers = hypers , optimizer_code = 0, p_star = 0)

                #Retrieve Original and Updated Factor Matrices
                Bk = lsr_ten.get_factor_matrix(s, k)
                Bk1 = weights
                if intercept: b = bias

                #Shape Bk1 as needed
                Bk1 = np.reshape(Bk1, (shape[k], ranks[k]), order = 'F') #if there is an error check here


                #Update Residuals and store updated factor matrix
                factor_residuals[s][k] = np.linalg.norm(Bk1 - Bk)
                updated_factor_matrices[s, k] = Bk1

                #Update Factor Matrix
                lsr_ten.update_factor_matrix(s, k, updated_factor_matrices[s, k])

                #update the intercept
                if intercept: lsr_ten.update_intercept(b)

                #Calculate Objective Function Value
                expanded_lsr = lsr_ten.expand_to_tensor()
                expanded_lsr = np.reshape(expanded_lsr, X[0].shape, order = 'F')
                objective_function_value = objective_function_tensor(y, X, expanded_lsr, lambda1)
                objective_function_values[iteration, s, k] = objective_function_value

                #Print Objective Function Values
                # print(f"Iteration: {iteration}, Separation Rank: {s}, Factor Matrix: {k}, Objective Function Value: {objective_function_value}")


        #Absorb necessary matrices into X, aside from core tensor, to get X_tilde
        #print('---------------------------------------------Core-------------------------------------------------')
        X, y = training_data, training_labels
        X_tilde, y_tilde = lsr_ten.bcd_core_update_x_y(X, y)

        #Solve the sub-problem pertaining to the core tensor
        hypers = {'lambda': lambda1, 'lr': lr, 'epochs': epochs, 'batch_size': 64, 'bias': intercept}
        weights, bias, loss_values, gap_to_optimality, nmse_values, corr_values, R2_values = SGD(X_tilde, y_tilde.reshape(-1,1), cost_function_code = 1, hypers = hypers, optimizer_code = 0, p_star = 0)

        #Get Original and Updated Core Tensor
        Gk = lsr_ten.get_core_matrix()
        Gk1 = np.reshape(weights, ranks, order = 'F')
        if intercept: b = bias

        #Update Residuals and store updated Core Tensor
        core_residual = np.linalg.norm(Gk1 - Gk)
        updated_core_tensor = Gk1

        #Update Core Tensor
        lsr_ten.update_core_matrix(updated_core_tensor)

        #Update Intercept

        if intercept: lsr_ten.update_intercept(b)

        #Calculate Objective Function Value
        if intercept: b = lsr_ten.get_intercept()
        expanded_lsr = lsr_ten.expand_to_tensor()
        expanded_lsr = np.reshape(expanded_lsr, X[0].shape, order = 'F')

        objective_function_value = objective_function_tensor(y, X, expanded_lsr, lambda1, b if intercept else None)
        objective_function_values[iteration, :, (len(ranks))] = objective_function_value
        #print('')
        #Print Objective Function Value
        # print(f"BCD Regression, Iteration: {iteration}, Core Tensor, Objective Function Value: {objective_function_value}")

        #Stopping Criteria
        diff = np.sum(factor_residuals.flatten()) + core_residual  #need to change this
        # print('------------------------------------------------------------------------------------------')
        # print(f"Value of Stopping Criteria: {diff}")
        # print(f"Expanded Tensor: {expanded_lsr}")
        # print('------------------------------------------------------------------------------------------')
        if diff < threshold: break


    return lsr_ten, objective_function_values