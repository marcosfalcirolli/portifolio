clear ; close all; clc
data = load('ex2data1.txt');
X = data(:, [1, 2]); y = data(:, 3);
[m, n] = size(X);
% Add intercept term to x and X_test
X = [ones(m, 1) X];
% Initialize fitting parameters
initial_theta = zeros(n + 1, 1);
g = zeros(size(X));

theta = initial_theta;
z=X*theta;

% for i = 1:m
%     for j = 1:n
%        g(i,j)=1/(1+exp(-(X(i,j))));
%        fprintf('\nValor de g(z): %f\n', g(i,j));
%     end
% end