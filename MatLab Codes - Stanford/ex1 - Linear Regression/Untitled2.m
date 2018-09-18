clear
clc

data = load('ex1data1.txt');
x = data(:, 1); y = data(:, 2);
m = length(y);
X = [ones(m, 1), data(:,1)]; % Add a column of ones to x
theta = zeros(2, 1); % initialize fitting parameters
soma = 0
somatoria = 0
for i=(1:m)
    X1 = X(i:i,1:2);
    yi = y(i:i,1:1)
    soma =((X1*theta)-yi)^2
    somatoria = soma + somatoria 
end
J = (1/(2*m))*somatoria