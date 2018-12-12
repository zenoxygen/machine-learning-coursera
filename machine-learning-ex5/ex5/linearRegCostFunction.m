function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% Compute regularized linear regression cost function
predictions = X * theta;
sqrErrors = (predictions - y) .^ 2;
J = 1 / (2 * m) * sum(sqrErrors);
reg = (lambda / (2 * m)) * sum(theta(2:end,:) .^ 2);
J = J + reg;

% Compute regularized linear regression gradient (vectorized version)
theta(1) = 0;
reg = (lambda / m) * theta;
errors = predictions - y;
grad = ((1 / m) * (X' * errors)) + reg;

% =========================================================================

grad = grad(:);

end
