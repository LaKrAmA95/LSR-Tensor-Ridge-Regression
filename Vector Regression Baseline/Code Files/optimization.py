## Import Optimization Toolkits
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

#Cost Function[Least Squares]
#||Xw - y ||_2^2
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
#||Xw - y ||_2^2 + lambda * ||w||^2_2
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
        l2_reg = 0
        for param in self.parameters():
            l2_reg += torch.norm(param) ** 2
        return self.lambda1 * l2_reg

#Optimize the Least Squares Cost Function via Stochastic Gradient Descent
#X: Shape n x d where n is the number of samples and d is the number of features
#Y: Shape n x 1 where n is the number of samples
#lr: learning rate
#epochs: number of epochs
#bias: whether we have intercept or not
def SGD1(X: np.ndarray, Y: np.ndarray, lr, epochs, batch_size, bias = False):
    #Initialize Cost Function
    cost_function = LeastSquares(X.shape[1])
    
    #Convert X and Y to pytorch tensors
    X = torch.tensor(X, dtype = torch.float32)
    Y = torch.tensor(Y, dtype = torch.float32)
    
    #Initialize SGD Optimizer
    optimizer = optim.SGD(cost_function.parameters(), lr = lr)
    
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

    if bias:
        return weights, cost_function.linear.bias.item(), loss_values
    else:
        return weights, 0, loss_values

#Optimize the Ridge Regression Cost Function via Stochastic Gradient Descent
#X: Shape n x d where n is the number of samples and d is the number of features
#Y: Shape n x 1 where n is the number of samples
#lamb: ridge parameter
#bias: whether we have intercept or not
def SGD2(X: np.ndarray, Y: np.ndarray, lamb, lr, epochs, batch_size, bias = False):
    #Initialize Cost Function
    cost_function = RidgeRegression(X.shape[1], lamb)
    
    #Convert X and Y to pytorch tensors
    X = torch.tensor(X, dtype = torch.float32)
    Y = torch.tensor(Y, dtype = torch.float32)
    
    #Initialize SGD Optimizer
    optimizer = optim.SGD(cost_function.parameters(), lr = lr)
    
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

    if bias:
        return weights, cost_function.linear.bias.item(), loss_values
    else:
        return weights, 0, loss_values