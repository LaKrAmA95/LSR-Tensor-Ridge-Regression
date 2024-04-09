import numpy as np
from optimization_PyTorch import objective_function_tensor
from sklearn.linear_model import SGDRegressor
import torch
import torch.nn as nn
import torch.optim as optim

#Cost Function[Least Squares + L2 Regularization Term]
#||X_tilde w - y ||_2^2 + lambda * ||w||^2_2
class CostFunction(nn.Module):
    def __init__(self, input_dim, lambda1, bias = False):
        super(CostFunction, self).__init__()
        self.linear = nn.Linear(input_dim, 1, bias) #input_dim = dimension of each sample in X_tilde, output_dim = 1 = number of y values for each sample
        self.lambda1 = lambda1 #Ridge Regression Lambda Value
        
    #Evaluate the Cost Function given x
    def evaluate(self, X_tilde, y_tilde):
        return nn.MSELoss(self.linear(X_tilde), y_tilde, reduction = 'sum') + self.l2_regularization()
            
    #Calculate value of lambda * ||w||^2_2
    def l2_regularization(self):
        l2_reg = 0
        for param in self.parameters():
            l2_reg += torch.norm(param) ** 2
        return self.lambda1 * l2_reg

def pytorch_subroutine(X_tilde: np.ndarray, y_tilde: np.ndarray, lambda1, bias = False):
    X_tilde_tensor = torch.tensor(X_tilde, dtype = torch.float32)
    y_tilde_tensor = torch.tensor(y_tilde, dtype = torch.float32)
    
    #Initialize Cost Function and the SGD Optimizer
    cost_function = CostFunction(X_tilde.shape[1], lambda1)
    optimizer = optim.SGD(cost_function.parameters(), lr=0.01)
    
    #For now, set batch size to 32 and number of epochs to 10
    batch_size = 32
    num_epochs = 10
    
    #Training Loop
    for epoch in range(num_epochs):
        # Shuffle dataset
        indices = torch.randperm(X_tilde_tensor.size(0))
        X_shuffled = X_tilde_tensor[indices]
        y_shuffled = y_tilde_tensor[indices]
        
        for i in range(0, X_shuffled.size(0), batch_size):
            # Get mini-batch
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            # Compute loss with L2 regularization
            loss = cost_function.evaluate(X_batch, y_batch)

            # Zero gradients
            optimizer.zero_grad()

            # Backward pass to compute gradient
            loss.backward()

            # Update parameters
            optimizer.step()

        # Print progress
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    weights = cost_function.linear.weight.numpy().flatten()
    
    if bias:
        return weights, cost_function.linear.bias.item()
    else:
        return weights, None #Return weights as numpy array
            

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
    b_intercept = intercept

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

                #Retrieve Original and Updated Factor Matrices
                Bk = lsr_ten.get_factor_matrix(s, k)
                Bk1, b = pytorch_subroutine(X_tilde, y_tilde, lambda1, intercept)

                #Shape Bk1 as needed
                Bk1 = np.reshape(Bk1, (shape[k], ranks[k]), order = 'F')

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
                objective_function_value = objective_function_tensor(y, X, expanded_lsr, lambda1)# CHECK THISSSS______________________________
                objective_function_values[iteration, s, k] = objective_function_value

                #Print Objective Function Values
                # print(f"Iteration: {iteration}, Separation Rank: {s}, Factor Matrix: {k}, Objective Function Value: {objective_function_value}")


        #Absorb necessary matrices into X, aside from core tensor, to get X_tilde
        X, y = training_data, training_labels
        X_tilde, y_tilde = lsr_ten.bcd_core_update_x_y(X, y)

        #Get Original and Updated Core Tensor
        Gk = lsr_ten.get_core_matrix()
        Gk1, b = pytorch_subroutine(X_tilde, y_tilde, lambda1, intercept)

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