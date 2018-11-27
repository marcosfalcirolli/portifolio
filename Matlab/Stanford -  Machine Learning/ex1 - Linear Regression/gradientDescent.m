function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples

J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    %theta0
    somatoria = sum((X*theta)-y);
    newtheta(1:1,1:1)=theta(1:1,1:1)-alpha*(1/m)*somatoria;
    %theta1
    somatoria = sum(((X*theta)-y).*X(:,2:2));
    newtheta(2:2,1:1)=theta(2:2,1:1)-alpha*(1/m)*somatoria;
    %updatingtheta
    theta=newtheta;
    J = computeCost(X, y, theta);
    %cheking
    fprintf('Testing Theta, Cost computed = %f\n', J);

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
