function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.


% the prediction vector
% theta = 2 x 1 --- transpose = 1 x 2
% X = m x 2     --- transpose = 2 x m
% H should be 1 x m
H = transpose(theta) * transpose(X);

% square difference
% SD should be m x 1 matrix (same as y)
SD = (transpose(H) - y) .^ 2;

% total difference
t = transpose(SD) * ones(m, 1);

J = (1 / (2 * m)) * t;

% =========================================================================

end
