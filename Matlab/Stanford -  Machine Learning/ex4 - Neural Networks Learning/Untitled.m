%% Initialization
clear ; close all; clc

%% Setup the parameters you will use for this exercise
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)
                          
load('ex4data1.mat');
m = size(X, 1);

% Load the weights into variables Theta1 and Theta2
load('ex4weights.mat');

% Unroll parameters 
nn_params = [Theta1(:) ; Theta2(:)];
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

a1 = [ones(m, 1) X];
z2 = (a1*Theta1');
a2 = sigmoid(z2);
a2 = [ones(m, 1) a2];
z3 = (a2*Theta2');
a3 = sigmoid(z3);

eye_matrix = eye(num_labels);
y_matrix = eye_matrix(y,:);

lambda = 0

J=(1/m)*sum(sum(-y_matrix.*log(a3)-(1-y_matrix).*log(1-a3)))+...
    (lambda/(2*m))*(...
    (sum(sum(Theta1(:,2:end).^2)))+...
    (sum(sum(Theta2(:,2:end).^2)))...
    );

delta3 = a3 - y_matrix;
delta2 = ((Theta2')*delta3')'.*((a2.*(1-a2)));

DELTA2 = zeros(size(Theta2))
DELTA2 = DELTA2 + ((delta3)'*a2)
DELTA1 = zeros(size(Theta1))
DELTA1 = DELTA1 + ((delta2(:,2:end))'*a1);

Theta1_grad(:,1) = 1/m * DELTA1(:,1)
Theta1_grad(:,2:end) = 1/m * DELTA1(:,2:end) + (lambda/m)*Theta1(:,2:end)
Theta2_grad(:,1) = 1/m * DELTA2(:,1)
Theta2_grad(:,2:end) = 1/m * DELTA2(:,2:end) + (lambda/m)*Theta2(:,2:end)