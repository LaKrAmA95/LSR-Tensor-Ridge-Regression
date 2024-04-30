import numpy as np
from scipy.io import loadmat

# Load the .mat file
mat_data = loadmat('features_rs.mat')

# Extract the data from the loaded .mat file
data = mat_data['features_rs']  # Replace 'variable_name' with the name of your variable

# Save as .npy file
np.save('fmri_rs.npy', data)
