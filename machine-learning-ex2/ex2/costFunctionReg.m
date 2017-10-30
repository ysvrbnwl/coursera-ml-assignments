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

hthetaX = sigmoid(X * theta);
loghthetaX = log(hthetaX);
otherlogthetaX = log(1 - hthetaX);

wholeTerm = ( transpose((-1 * y)) * loghthetaX ) - ( transpose((1 - y)) * otherlogthetaX );
thetaTranpose = transpose(theta);
regu = ((lambda/(2*m)) * sum(thetaTranpose(1,2:length(thetaTranpose)).^2));
J = ((1/m) * wholeTerm) + regu;
subtr = hthetaX - y;
a = transpose((1/m) .* (transpose(subtr)*X));
b = transpose((lambda/m).* transpose(theta));

grad(1,1) = a(1,1);
grad(2:length(grad), 1) = a(2:length(grad), 1) + b(2:length(grad), 1);
% =============================================================

end
