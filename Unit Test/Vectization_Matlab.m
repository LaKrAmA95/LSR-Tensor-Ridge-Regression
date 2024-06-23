%% MATRIX TO VECTOR 

% Define matrix A
A = [1  2  3  4; 2  1  7  8; 3 7 1 12; 4 8 12 1]

% Masking the upper triangle
mask = triu(true(size(A)), 1);  % Masking upper triangle (excluding diagonal)

% Extracting the elements of the upper triangle
upperTriangleVector = A(mask)

% Displaying the vector
disp('Upper triangle vector:');
disp(upperTriangleVector);


%vectorization 
A_vectorized = A(:);



%matlab vectorization happens column-wise


%% Vector to Matrix

%let's try to reconstruct the symmetric matrix from [2,3,4,5,6,7]

output_dim = 4

%FC values
vector = [2 3 4 5 6 7]
%Upper Triangle Indices
upper_triangle_indices = triu(true(output_dim),1);
%Sample Dummy
samples = eye(output_dim,output_dim);
%Upper triangle filled dummy
samples(upper_triangle_indices) = vector;
% Symmetrize the matrix
samples = samples + samples' - diag(diag(samples));

samples
disp(upper_triangle_indices)

%% HCP to Matrix 

% Specify the filename
filename = ['X_train_vec.csv']; % Replace with your actual file name

% Read the CSV file
data = readmatrix(filename);

%matrix 
matrix = data(1,:);
%output dimnesion
output_dim = 400;
%Upper Triangle Indices
upper_triangle_indices = triu(true(output_dim),1);
%Sample Dummy
samples = eye(output_dim,output_dim);
%Upper triangle filled dummy
samples(upper_triangle_indices) = matrix;
% Symmetrize the matrix
samples = samples + samples' - diag(diag(samples));


% Generate the heatmap
heatmap(samples);

% Optionally, customize the heatmap
title('Sample Heatmap');
xlabel('X-axis Label');
ylabel('Y-axis Label');



