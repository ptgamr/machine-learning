function [error_train, error_val] = ...
    learningCurveRandom(X, y, Xval, yval, lambda)

% Number of training examples
m = size(X, 1);
r = size(Xval, 1);

% You need to return these values correctly
error_train = zeros(m, 1);
error_val   = zeros(m, 1);

mval = size(Xval, 1);

for i = 1:m
  Jtrain = zeros(50, 1);
  Jval = zeros(50, 1);
  for j = 1:50
    M_rand_idx  = randperm(m);
    R_rand_idx = randperm(r);

    Xrand = X(M_rand_idx(1:i), :);
    yrand = y(M_rand_idx(1:i), :);
    Xval_rand = Xval(R_rand_idx(1:i), :);
    yval_rand = yval(R_rand_idx(1:i), :);

    [theta] = trainLinearReg(Xrand, yrand, lambda);
    Jtrain(j) = linearRegCostFunction(Xrand, yrand, theta, 0);
    Jval(j) = linearRegCostFunction(Xval_rand, yval_rand, theta, 0);
  end

  error_train(i) = mean(Jtrain);
  error_val(i) = mean(Jval);
end

% -------------------------------------------------------------

% =========================================================================

end
