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

X = [ones(m, 1) X]; % 5000 x 401

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

a2 = sigmoid(X * Theta1'); % m x 401 * 401 x 25 = m x 25
a2 = [ones(m, 1) a2]; % m x 26
a3 = sigmoid(a2 * Theta2'); % m x 26 * 26 * 10 = m * 10

% convert Y to matrix of 0 & 1
% size(Y) = m x 10
Y = zeros(m, num_labels);

for i = 1:m
  y_at_i = y(i, 1);
  Y(i, y_at_i) = 1;
endfor

% lrCostFunction
% J = -(1/m) * (y' * log(h) + (1-y)' * log(1-h)) + (lambda / (2 * m)) * sum((theta_for_regularization' .^ 2));

% 10 x m * m * 10 = 10 * 10
%J = -1/m * (Y' * log(a3) + (1-Y)' * log(1-a3));

J = 0;
for c = 1:m
  Y_at_c = Y(c, :); % 1 x 10
  a3_at_c = a3(c, :); % 1 x 10

  %  10 x 1 * 1 x 10 = 10 x 10
  J += -1/m * (Y_at_c * log(a3_at_c)' + (1 - Y_at_c) * log(1-a3_at_c)');
endfor

% Regularlize cost function
Theta1_reg = Theta1;
Theta1_reg(: , 1) = 0; % set all of first column to 0, as we don't regularize the bias

Theta2_reg = Theta2;
Theta2_reg(: , 1) = 0;

J += (lambda/(2*m)) * (sum((Theta1_reg' .^ 2)(:)) + sum((Theta2_reg' .^ 2)(:)));



% Backpropagation

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
