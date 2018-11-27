function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
J=(1/m)*sum(-y.*log(sigmoid(X*theta))-(1-y).*log(1-sigmoid(X*theta)));
%grad = 1/m*sum()

    %theta0
    %somatoria = sum((sigmoid(X*theta))-y);
    %grad(1:1,1:1)=(1/m)*somatoria;
    grad(1:1,1:1)=(1/m)*sum(((sigmoid(X*theta))-y).*X(:,1:1));
    %theta1
    somatoria = sum(((sigmoid(X*theta))-y).*X(:,2:2));
    grad(2:2,1:1)=(1/m)*somatoria;
    %theta2
    somatoria = sum(((sigmoid(X*theta))-y).*X(:,3:3));
    grad(3:3,1:1)=(1/m)*somatoria;
    

% =============================================================

end