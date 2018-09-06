function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% ==========================================================
% Part 1: Forward propagation and cost function
% ==========================================================

% Add ones to the X data matrix
X = [ones(m, 1) X];

% Transpose X to get a column vector
a1 = X';

% Computes z2 then a2
z2 = Theta1 * a1;
a2 = sigmoid(z2);

% Transpose a2 to correctly add the extra bias unit to the layer
a2 = a2';
a2 = [ones(size(a2, 1), 1) a2];
a2 = a2';

% Compute z3 then a3
z3 = Theta2 * a2;
a3 = sigmoid(z3);

% Recode the labels as vectors containing only values 0 or 1
% Before the operation, y is 5000x1 and after, y is 10x5000
y_recoded = zeros(num_labels, m);
for i = 1:m
	y_recoded(y(i), i) = 1;
end

% Transpose the a3 matrix to correctly compute the predictions
a3 = a3';

% h_theta is a matrix containing the output layer
h_theta = a3;

% Excluding the bias term of Theta1 and Theta2
theta1 = Theta1(:,2:end);
theta2 = Theta2(:,2:end);

% Calcul the regularization term
reg = (lambda / (2 * m)) * (sum(sum(theta1 .^ 2)) + sum(sum(theta2 .^ 2)));

% Compute cost function
J = (1 / m) * sum(sum(-y_recoded' .* log(h_theta) - (1 - y_recoded)' .* log(1 - h_theta))) + reg;

% ==========================================================
% Part 2: Backpropagation
% ==========================================================

for t = 1:m
	% Step 1: perform a feedforward pass
	% Set the input layer's value to the t-th training example
	a1 = X(t,:);
	% Transpose a1 to get a column vector
	a1 = a1';
	% Computes z2 then a2
	z2 = Theta1 * a1;
	a2 = sigmoid(z2);
	% Transpose a2 to add the bias unit
	a2 = a2';
	a2 = [1 a2];
	a2 = a2';
	% Compute z3 then a3
	z3 = Theta2 * a2;
	a3 = sigmoid(z3);

	% Step 2: compute error values for layer 3
	delta3 = a3 - y_recoded(:,t);

	% Step 3: compute error values for layer 2
	delta2 = (Theta2' * delta3) .* a2 .* (1 - a2);

	% Step 4: accumulate the gradient
	% Remove first unit of delta_2
	delta2 = delta2(2:end);
	% Update our Delta matrix (vectorized version)
	Theta1_grad = Theta1_grad + delta2 * a1';
	Theta2_grad = Theta2_grad + delta3 * a2';
end

% Step 5: obtain the unregularized gradient for the neural network cost function
% Divide the accumulated gradients by 1/m
Theta1_grad = Theta1_grad * (1 / m);
Theta2_grad = Theta2_grad * (1 / m);

% =========================================================================

% Regularization
Theta1_grad = Theta1_grad + (lambda / m) * [zeros(size(Theta1, 1), 1) Theta1(:,2:end)];
Theta2_grad = Theta2_grad + (lambda / m) * [zeros(size(Theta2, 1), 1) Theta2(:,2:end)];

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];
