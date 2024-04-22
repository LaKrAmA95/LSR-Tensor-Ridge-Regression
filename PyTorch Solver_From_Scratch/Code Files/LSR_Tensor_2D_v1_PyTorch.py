import numpy as np
import torch

# Low Separation Rank Tensor Decomposition
class LSR_tensor_dot():
    # Constructor
    def __init__(self, shape, ranks, separation_rank, dtype = torch.float32 ,initialize = True, device = torch.device("cpu")):
        super(LSR_tensor_dot, self).__init__()
        self.shape = shape
        self.ranks = ranks
        self.separation_rank = separation_rank
        self.dtype = dtype
        self.device = device
        self.order = len(shape)
        self.init_params(initialize)

    # Initialize Parameters
    def init_params(self,initialize = True):
        # Initialize core tensor as independent standard gaussians
        if not initialize:
            #self.core_tensor = np.zeros(shape = self.ranks)
            self.core_tensor = torch.nn.parameter.Parameter(torch.zeros(self.ranks, device = self.device))
        else:
            #self.core_tensor = np.random.normal(size = self.ranks)
            self.core_tensor = torch.nn.parameter.Parameter(torch.normal(mean=0,std=1,size=self.ranks,device=self.device))
            
        # Set up Factor Matrices
        
        #self.factor_matrices = []
        self.factor_matrices = torch.nn.ModuleList()
        
        
        # Initialize all factor matrices
        for s in range(self.separation_rank):
            factors_s = torch.nn.ParameterList()
            for k in range(self.order):
                if not initialize:
                    #factor_matrix_B = np.eye(self.shape[k])[:, self.ranks[k]]
                    factor_matrix_B = torch.eye(self.shape[k])[:,self.rank[k]]
                    factors_s.append(factor_matrix_B)
                else:
                    #factor_matrix_A = np.random.normal(0,1,size= (self.shape[k], self.ranks[k]))
                    factor_matrix_A = torch.normal(mean=0,std=1,size=[self.shape[k],self.ranks[k]],dtype=self.dtype, device = self.device)
                    factors_s.append(factor_matrix_A)

            self.factor_matrices.append(factors_s)


    # Expand core tensor and factor matrices to full tensor, optionally excluding
    # a given term from the separation rank decomposition
    
    def expand_to_tensor(self, skip_term = None):
        full_lsr_tensor = torch.zeros(size = self.shape)

        #Calculate Expanded Tensor
        for s, term_s_factors in enumerate(self.factor_matrices):
            if s == skip_term: continue
            expanded_tucker_term = term_s_factors[0] @ self.core_tensor @ term_s_factors[1].T
            full_lsr_tensor += expanded_tucker_term

        #Column Wise Flatten full_lsr_tensor
        #full_lsr_tensor = full_lsr_tensor.flatten(order = 'F')
        full_lsr_tensor = full_lsr_tensor.T.flatten() # the transposing make the column major flattening possible
        return full_lsr_tensor

    # Absorb all factor matrices and core tensor into the input tensor except for matrix s, k
    # Used during a factor matrix update step of block coordiante descent

    def bcd_factor_update_x_y(self, s, k, x, y):
        # Convert inputs to PyTorch tensors if they are not already
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        # Take x and swap axes 1 and 2 so that vectorization occurs "COLUMN WISE"
        x_transpose = x.transpose(1, 2)
        x_transpose_vectorized = x_transpose.reshape(x_transpose.shape[0], -1)

        # If we are unfolding along mode 1, use x. Else, if we are unfolding along mode, use x_transpose
        x_partial_unfold = x if k == 0 else x_transpose

        # If k = 0 (skip first factor matrix), we have 2nd factor matrix. If k = 1 (skip second factor matrix), we have first factor matrix
        kronecker_term = self.factor_matrices[s][1] if k == 0 else self.factor_matrices[s][0]

        # If k = 0, G^T. Else if k = 1, put G
        core_tensor_term = self.core_tensor.T if k == 0 else self.core_tensor

        omega = x_partial_unfold @ kronecker_term @ core_tensor_term
        omega = omega.transpose(1, 2)
        omega = omega.reshape(omega.shape[0], -1)

        X_tilde = omega
        y_tilde = y

        if self.separation_rank != 1:
            gamma = x_transpose_vectorized @ self.expand_to_tensor(skip_term=s)
            y_tilde = y - gamma

        return X_tilde, y_tilde

    # Absorb all factor matrices the input tensor (not the core tensor)
    # Used during a core tensor update step of block coordiante descent
    
    def bcd_core_update_x_y(self, x, y):

      
        #converting into torch tensor 
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        
        # Take x and swap axes 1 and 2 so that vectorization occurs "COLUMN WISE"
        x_transpose = x.transpose(1, 2)
        x_transpose_vectorized = x_transpose.reshape(x_transpose.shape[0], -1)

        # Calculate y_tilde (assuming no change needed, direct assignment)
        y_tilde = y

        # Calculate Kronecker Factor Sum
        kron_factor_sum = 0
        
        for term_s_factors in self.factor_matrices:
        # Calculate Kronecker product using torch.kron
        # Note: torch.kron is generally not available in older versions of PyTorch, update if necessary
            kron_product = torch.kron(term_s_factors[1], term_s_factors[0])
            kron_factor_sum += kron_product

        # Perform matrix multiplications and transpose to get the final shape
        # Note: The transposition sequence here mimics numpy behavior
        core_update = (kron_factor_sum.T @ x_transpose_vectorized.T).T

        return core_update, y_tilde


    #Retrieve factor matrix
    def get_factor_matrix(self, s, k):
      return self.factor_matrices[s][k]

    #Update factor matrix
    #def update_factor_matrix(self, s, k, updated_factor_matrix):
    #  self.factor_matrices[s][k] = updated_factor_matrix

    #Retrieve Core Matrix
    def get_core_matrix(self):
      return self.core_tensor

    #Update core matrix
    #def update_core_matrix(self, updated_core_matrix: np.ndarray):
    #  self.core_tensor = updated_core_matrix

