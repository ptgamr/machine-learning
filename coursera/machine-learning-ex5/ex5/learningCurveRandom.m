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
    M_rand  = randperm(m)
    R_rand = randperm(r)

    Xrand = M_rand(1:i)
    Xval_rand = R_rand(1:i)
  end


  Xtrain = X(1:i, :);
  ytrain = y(1:i);
  [theta] = trainLinearReg(Xtrain, ytrain, 1);

  [J] = linearRegCostFunction(Xtrain, ytrain, theta, 0);
  error_train(i) = J;

  [J] = linearRegCostFunction(Xval, yval, theta, 0);
  error_val(i) = J;
end







% -------------------------------------------------------------

% =========================================================================

end
