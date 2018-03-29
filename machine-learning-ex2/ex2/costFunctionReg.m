function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% Calcul h
h = sigmoid(X * theta);
% Calcul the regularization term by excluding the bias term theta_0
reg = (lambda / (2 * m)) * sum(theta(2:end,:) .^ 2);
% Compute cost function by adding the regularization term (vectorized version)
J = (1 / m) * (-y' * log(h) - (1 - y)' * log(1 - h)) + reg;

% Calcul the regularization term
reg = (lambda / m) * theta(2:end,:);
% Set the first lign of the vector to 0 since we don't want to penalize theta_0
reg = [0; reg];
% Compute gradient descent by adding the regularization term
num_iters = size(theta);
predictions = h;
errors = (predictions - y);
for iter = 1:num_iters
  grad(iter) = ((1 / m) * sum(errors .* X(:,iter))) + reg(iter);
end

% =============================================================

end
