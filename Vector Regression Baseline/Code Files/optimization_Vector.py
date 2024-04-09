## Contains All Helper Functions for Optimization
import numpy as np
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
        mse_loss = nn.MSELoss(reduction = 'sum')
        return mse_loss(self.linear(X_tilde), y_tilde) + self.l2_regularization()
            
    #Calculate value of lambda * ||w||^2_2
    def l2_regularization(self):
        l2_reg = 0
        for param in self.parameters():
            l2_reg += torch.norm(param) ** 2
        return self.lambda1 * l2_reg

#Optimize the Cost Function via Stochastic Gradient Descent
#X_train: Shape n x d where n is the number of samples and d is the number of features
#Y_train: Shape n x 1 where n is the number of samples
#lambda1: ridge parameter
def optimization_subroutine(X_train: np.ndarray, Y_train: np.ndarray, lambda1, bias = False):
    X_train_tensor = torch.tensor(X_train, dtype = torch.float32)
    Y_train_tensor = torch.tensor(Y_train, dtype = torch.float32)

    #Initialize Cost Function and the SGD Optimizer
    cost_function = CostFunction(X_train.shape[1], lambda1)
    optimizer = optim.SGD(cost_function.parameters(), lr = 0.0001)

    #For now, set batch size to 256 and number of epochs to 10
    batch_size = 256
    num_epochs = 100

    #Training Loop
    for epoch in range(num_epochs):
        # Shuffle dataset
        indices = torch.randperm(X_train_tensor.size(0))
        X_shuffled = X_train_tensor[indices]
        y_shuffled = Y_train_tensor[indices]

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
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    weights = cost_function.linear.weight.data.numpy().reshape((-1, 1)) #Return weights as numpy array

    if bias:
        return weights, cost_function.linear.bias.item()
    else:
        return weights, 0