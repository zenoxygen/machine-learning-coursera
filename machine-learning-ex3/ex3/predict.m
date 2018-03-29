function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% =========================
% Computes the hidden layer
% =========================

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

% =========================
% Computes the output layer
% =========================

% Computes z3 then a3
z3 = Theta2 * a2;
a3 = sigmoid(z3);

% Transpose the matrix a3 to correctly computes the predictions
a3 = a3';

% Computes the predictions for the output layer a3
[max_elem, idx_max_elem] = max(a3, [], 2);

% Put in vector p the index of the max element for each row
p = idx_max_elem;

% =========================================================================

end
