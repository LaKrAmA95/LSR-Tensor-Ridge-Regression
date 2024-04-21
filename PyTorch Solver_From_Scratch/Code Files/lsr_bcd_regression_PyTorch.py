import numpy as np
from optimization_PyTorch import objective_function_tensor
from sklearn.linear_model import SGDRegressor
import torch
import torch.nn as nn
import torch.optim as optim

#Cost Function[Least Squares + L2 Regularization Term]
#||X_tilde w - y ||_2^2 + lambda * ||w||^2_2

class CostFunction(nn.Module):
    def __init__(self, lsr_ten, lambda1):
        super(CostFunction, self).__init__()
        
        self.lambda1 = lambda1 #Ridge Regression Lambda Value
        self.active_param = None
    
    #setting the parameter that we are going to update.
    def set_active_param(self,param):
        self.active_param = param
    
    #the forward step based on each of subproblem    
    def forward(self,x):
        if self.active_param == 'Core':
            return torch.matmul(x,lsr_ten.core_tensor.T.flatten())
        elif self.active_param is not None:
            return torch.matmul(x,lsr_ten.factor_matrices[self.active_param[0]][self.active_param[1]].T.flatten())
        raise ValueError("No active parameter set for forward computation.")
    
                
    def l2_regularization(self):
        l2_reg = 0
        if self.active_param == 'Core':
            l2_reg = torch.norm(lsr_ten.core_tensor)**2
            return self.lambda1 * l2_reg
        elif self.active_param is not None:
            l2_reg = torch.norm(lsr_ten.factor_matrices[self.active_param[0]][self.active_param[1]])**2
            return self.lambda1 * l2_reg
        raise ValueError("No active parameter set for forward computation.")
        
    #Evaluate the Cost Function given x
    def evaluate(self, X_tilde, y_tilde):
        mse_loss = nn.MSELoss(reduction = 'sum')
        print(f"Predicted: {self.forward(X_tilde)}, Actual: {y_tilde}")
        return mse_loss(self.forward(X_tilde), y_tilde) + self.l2_regularization()

# ----------------------------------------------------------------------------------End of Cost Function-----------------------------------------------------------#            

#lsr_ten: LSR Tensor
#training_data: X
#training_labels: Y
#hypers: hyperparameters

def lsr_bcd_regression(lsr_ten, training_data: np.ndarray, training_labels: np.ndarray, hypers: dict,):
    #Get LSR Tensor Information and other hyperparameters
    shape, ranks, sep_rank, order = lsr_ten.shape, lsr_ten.ranks, lsr_ten.separation_rank, lsr_ten.order
    lambda1 = hypers["weight_decay"]
    max_iter = hypers["max_iter"]
    threshold = hypers["threshold"]
    #b_intercept = intercept

    #initializing all the solvers
    
    factor_matrix_models = [[optim.SGD([lsr_ten.factor_matrices[s][k]],lr=0.0001) for k in range(len(ranks))] for s in range(sep_rank)]
    core_tensor_model = optim.SGD([lsr_ten.core_tensor], lr = 0.0001)
    
    #constructing the model 
    
    Cost_Model = CostFunction(lsr_ten,lambda1)
    
    #Store objective function values
    objective_function_values = np.ones(shape = (max_iter, sep_rank, len(ranks) + 1)) * np.inf

    #Normalized Estimation Error
    iterations_normalized_estimation_error = np.zeros(shape = (max_iter,))

    #Run at most max_iter iterations of Block Coordinate Descent
    for iteration in range(max_iter):
        factor_residuals = np.zeros(shape = (sep_rank, len(ranks)))
        core_residual = 0

        #Store updates to factor matrices and core tensor
        updated_factor_matrices = np.empty((sep_rank, len(ranks)), dtype=object)
        updated_core_tensor = None

        #Iterate over the Factor Matrices.
        for s in range(sep_rank):
            for k in range(len(ranks)):
                #Absorb Factor Matrices into X aside from (s, k) to get X_tilde
                X, y = training_data, training_labels
                X_tilde, y_tilde = lsr_ten.bcd_factor_update_x_y(s, k, X, y) #y tilde should now be y-b-<Q,X>                

                #holding the old parameter
                Bk = lsr_ten.factor_matrices[s][k].detach().cpu().numpy()
                
                #Retrieve Original and Updated Factor Matrices
                
                Cost_Model.active_param([s,k])
                factor_matrix_models[s][k].zero_grad()
                loss_factor = Cost_Model.evaluate(X_tilde,y_tilde,)
                loss_factor.backward()
                factor_matrix_models[s][k].step()
                
                #getting 
                Bk_1 = lsr_ten.factor_matrices[s][k].detach().cpu().numpy()
                #Shape Bk1 as needed
                Bk1 = np.reshape(Bk1, (shape[k], ranks[k]), order = 'F')

                #Update Residuals and store updated factor matrix
                factor_residuals[s][k] = np.linalg.norm(Bk1 - Bk)
                updated_factor_matrices[s, k] = Bk1

                #Update Factor Matrix
                #lsr_ten.update_factor_matrix(s, k, updated_factor_matrices[s, k])

                #update the intercept
                #if intercept: lsr_ten.update_intercept(b)

                #Calculate Objective Function Value
                expanded_lsr = lsr_ten.expand_to_tensor().detach().cpu().numpy()
                expanded_lsr = np.reshape(expanded_lsr, X[0].shape, order = 'F')
                objective_function_value = objective_function_tensor(y, X, expanded_lsr, lambda1)
                objective_function_values[iteration, s, k] = objective_function_value

                #Print Objective Function Values
                # print(f"Iteration: {iteration}, Separation Rank: {s}, Factor Matrix: {k}, Objective Function Value: {objective_function_value}")


        #Absorb necessary matrices into X, aside from core tensor, to get X_tilde
        X, y = training_data, training_labels
        X_tilde, y_tilde = lsr_ten.bcd_core_update_x_y(X, y)

        #Get Original and Updated Core Tensor
        Gk = lsr_ten.get_core_matrix().detach().cpu().numpy()
        
        #core tensor update test 
        
        Cost_Model.set_active_param('Core')
        core_tensor_model.zero_grad()
        loss_core = CostFunction.evaluate(X_tilde,y_tilde)
        loss_core.backward() 
        core_tensor_model.step()
        
        Gk1 = lsr_ten.get_core_matrix().detach().cpu().numpy()
        
        #Reshape Gk1
        Gk1 = np.reshape(Gk1, ranks, order = 'F')

        #Update Residuals and store updated Core Tensor
        core_residual = np.linalg.norm(Gk1 - Gk)
        updated_core_tensor = Gk1

        #Update Core Tensor
        lsr_ten.update_core_matrix(updated_core_tensor)

        #Update Intercept
        #if intercept: lsr_ten.update_intercept(b)

        #Calculate Objective Function Value
        #if intercept: b = lsr_ten.get_intercept()
        expanded_lsr = lsr_ten.expand_to_tensor().detach().cpu().numpy()
        expanded_lsr = np.reshape(expanded_lsr, X[0].shape, order = 'F')

        objective_function_value = objective_function_tensor(y, X, expanded_lsr, lambda1)
        objective_function_values[iteration, :, (len(ranks))] = objective_function_value

        # Print Objective Function Value
        # print(f"BCD Regression, Iteration: {iteration}, Core Tensor, Objective Function Value: {objective_function_value}")

        #Stopping Criteria
        diff = np.sum(factor_residuals.flatten()) + core_residual  #need to change this
        # print('------------------------------------------------------------------------------------------')
        # print(f"Value of Stopping Criteria: {diff}")
        # print(f"Expanded Tensor: {expanded_lsr}")
        # print('------------------------------------------------------------------------------------------')
        if diff < threshold: break

    return lsr_ten, objective_function_values
