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
n = size(X, 2);
h = hidden_layer_size;
r = num_labels;

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

a1 = X;
z2 = X * Theta1';
a2 = sigmoid(z2); % m x 401 * 401 x 25 = m x 25
a2 = [ones(m, 1) a2]; % m x 26
z3 = a2 * Theta2';
a3 = sigmoid(z3); % m x 26 * 26 * 10 = m * 10

% convert Y to matrix of 0 & 1
% size(Y) = m x 10
y_matrix =eye(num_labels)(y, :);


% Have to use element wise product (as matrix multiply will add unwanted term)
J = -1/m * sum(sum((y_matrix .* log(a3) + (1 - y_matrix) .* log(1 - a3))));


% Regularlize cost function
Theta1_reg = Theta1;
Theta1_reg(: , 1) = 0; % set all of first column to 0, as we don't regularize the bias

Theta2_reg = Theta2;
Theta2_reg(: , 1) = 0;

J += (lambda/(2*m)) * (sum((Theta1_reg' .^ 2)(:)) + sum((Theta2_reg' .^ 2)(:)));


% Backpropagation

% --------------- FOR LOOP IMPLEMENTATION -----------

Delta1 = zeros(h, n);
Delta2 = zeros(r, h+1);

%for t = 1:m
%  x = X(t, :); % 1x401
%
%  a1 = x';
%
%  z2 = Theta1 * a1; % 25x401 * 401x1 = 25x1
%  a2 = sigmoid(z2);
%  a2 = [1; a2]; % 26x1
%
%  z3 = Theta2 * a2; % 10x26 * 26x1 = 10x1
%  a3 = sigmoid(z3); % 10x1
%
%  d3 = a3 - Y(t, :)'; % 10x1
%
%  d2 = ((Theta2(:, 2:end))' * d3) .* sigmoidGradient(z2); % 25x10 * 10x1 .* 25x1
%
%  Delta1 += d2 * a1'; % 25x1 * 1x401
%  Delta2 += d3 * a2'; % 10x1 * 1x26 = 10x26
%endfor

% --------------- VECTORIZED IMPLEMENTATION -----------

d3 = a3 - y_matrix; % mx10
d2 = (d3 * Theta2(:, 2:end)) .* sigmoidGradient(z2); % mx25

Delta1 = d2' * a1; % 25xm * mx401
Delta2 = d3' * a2; % 10xm * mx26

Theta1_grad = Delta1 ./ m;
Theta2_grad = Delta2 ./ m;

% Add regularization

Theta1_grad += (lambda/m) * Theta1_reg;
Theta2_grad += (lambda/m) * Theta2_reg;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
