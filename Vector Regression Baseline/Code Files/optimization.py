## Import Optimization Toolkits
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

#Cost Function[Least Squares]
#1/n ||(XW + b) - Y||_2^2
class LeastSquares(nn.Module):
    def __init__(self, input_dim, bias = False):
        super(LeastSquares, self).__init__() #Initialize class
        self.linear = nn.Linear(input_dim, 1, bias) #input_dim = dimension of each sample in X, output_dim = 1 = number of y values for each sample
        nn.init.normal_(self.linear.weight) #Initialize weights to be from standard normal distribution
        if bias:
            nn.init.normal_(self.linear.bias) #Initialize bias to be from standard normal distribution
        
    #Evaluate the Cost Function given X and y
    def evaluate(self, X, y, reduction = 'mean'):
        mse_loss = nn.MSELoss(reduction = reduction)
        return mse_loss(self.linear(X), y)

#Cost Function[Least Squares + L2 Regularization Term]
#1/n ||(XW + b) - Y ||_2^2 + lambda * ||w||^2_2
class RidgeRegression(nn.Module):
    def __init__(self, input_dim, lambda1, bias = False):
        super(RidgeRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1, bias) #input_dim = dimension of each sample in X, output_dim = 1 = number of y values for each sample
        nn.init.normal_(self.linear.weight) #Initialize weights to be from standard normal distribution
        if bias:
            nn.init.normal_(self.linear.bias) #Initialize bias to be from standard normal distribution
        self.lambda1 = lambda1 #Ridge Regression Lambda Value
        
    #Evaluate the Cost Function given X and y
    def evaluate(self, X, y, reduction = 'mean'):
        mse_loss = nn.MSELoss(reduction = reduction)
        return mse_loss(self.linear(X), y) + self.l2_regularization()
            
    #Calculate value of lambda * ||w||^2_2
    def l2_regularization(self):
        return self.lambda1 * (torch.norm(self.linear.weight) ** 2)

#Optimize the Least Squares Cost Function via Stochastic Gradient Descent
#X: Shape n x d where n is the number of samples and d is the number of features
#Y: Shape n x 1 where n is the number of samples
#lr: learning rate
#epochs: number of epochs
#batch_size: batch size
#momentum: momentum
#dampening: dampnent constant
#nesterov: True to Enable Nesterov Momentum Computation
#decay_factor: Decay Factor for RMSProp
#optimizer_code: 0 for SGD, 1 for Adagrad, 2 for RMSProp
#bias: whether we have intercept or not
def SGD1(X: np.ndarray, Y: np.ndarray, lr = 0.005, epochs = 500, batch_size = 1, momentum = 0, dampening = 0, nesterov = False, decay_factor = 0, optimizer_code = 0, bias = False):
    #Initialize Cost Function
    cost_function = LeastSquares(X.shape[1], bias)
    
    #Convert X and Y to pytorch tensors
    X = torch.tensor(X, dtype = torch.float32)
    Y = torch.tensor(Y, dtype = torch.float32)
    
    #Initialize Optimizer
    optimizer = None
    if optimizer_code == 0:
        optimizer = optim.SGD(cost_function.parameters(), lr = lr, momentum = momentum, dampening = dampening, nesterov = nesterov)
    elif optimizer_code == 1:
        optimizer = optim.Adagrad(cost_function.parameters(), lr = lr)
    elif optimizer_code == 2:
        optimizer = optim.RMSprop(cost_function.parameters(), lr = lr, alpha = decay_factor, momentum = momentum)
    
    #Store batch loss values
    loss_values = []

    #Training Loop
    for epoch in range(epochs):
        # Shuffle dataset
        indices = torch.randperm(X.size(0))
        X_shuffled = X[indices]
        y_shuffled = Y[indices]
        
        #Get X and Y sample
        X_sample = X_shuffled[0: batch_size]
        Y_sample = y_shuffled[0: batch_size]
            
        # Compute stochastic loss
        stochastic_loss = cost_function.evaluate(X_sample, Y_sample, 'mean')

        # Zero gradients
        optimizer.zero_grad()

        # Backward pass to compute stochastic gradient
        stochastic_loss.backward()

        # Update parameters
        optimizer.step()
                
        #Print and Store batch loss values
        batch_loss = cost_function.evaluate(X, Y, 'mean')
        loss_values.append(batch_loss.item())
        
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {batch_loss:.4f}')

    weights = cost_function.linear.weight.data.numpy().reshape((-1, 1)) #Return weights as numpy array

    #return weights and bias
    if bias:
        return weights, cost_function.linear.bias.item(), loss_values
    else:
        return weights, 0, loss_values

#Optimize the Ridge Regression Cost Function via Stochastic Gradient Descent
#X: Shape n x d where n is the number of samples and d is the number of features
#Y: Shape n x 1 where n is the number of samples
#lamb: ridge parameter
#lr: learning rate
#epochs: number of epochs
#batch_size: batch size
#momentum: momentum
#dampening: dampnent constant
#nesterov: True to Enable Nesterov Momentum Computation
#decay_factor: Decay Factor for RMSProp
#optimizer_code: 0 for SGD, 1 for Adagrad, 2 for RMSProp
#bias: whether we have intercept or not
def SGD2(X: np.ndarray, Y: np.ndarray, lamb = 0.1, lr = 0.005, epochs = 500, batch_size = 1, momentum = 0, dampening = 0, nesterov = False, decay_factor = 0, optimizer_code = 0, bias = False):
    #Initialize Cost Function
    cost_function = RidgeRegression(X.shape[1], lamb, bias)
    
    #Convert X and Y to pytorch tensors
    X = torch.tensor(X, dtype = torch.float32)
    Y = torch.tensor(Y, dtype = torch.float32)
    
    #Initialize Optimizer
    optimizer = None
    if optimizer_code == 0:
        optimizer = optim.SGD(cost_function.parameters(), lr = lr, momentum = momentum, dampening = dampening, nesterov = nesterov)
    elif optimizer_code == 1:
        optimizer = optim.Adagrad(cost_function.parameters(), lr = lr)
    elif optimizer_code == 2:
        optimizer = optim.RMSprop(cost_function.parameters(), lr = lr, alpha = decay_factor, momentum = momentum)
    
    #Store batch loss values
    loss_values = []

    #Training Loop
    for epoch in range(epochs):
        # Shuffle dataset
        indices = torch.randperm(X.size(0))
        X_shuffled = X[indices]
        y_shuffled = Y[indices]
        
        #Get X and Y sample
        X_sample = X_shuffled[0: batch_size]
        Y_sample = y_shuffled[0: batch_size]
            
        # Compute stochastic loss
        stochastic_loss = cost_function.evaluate(X_sample, Y_sample, 'mean')

        # Zero gradients
        optimizer.zero_grad()

        # Backward pass to compute stochastic gradient
        stochastic_loss.backward()

        # Update parameters
        optimizer.step()
                
        #Print and Store batch loss values
        batch_loss = cost_function.evaluate(X, Y, 'mean')
        loss_values.append(batch_loss.item())
        
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {batch_loss:.4f}')

    weights = cost_function.linear.weight.data.numpy().reshape((-1, 1)) #Return weights as numpy array

    #return weights and bias
    if bias:
        return weights, cost_function.linear.bias.item(), loss_values
    else:
        return weights, 0, loss_values