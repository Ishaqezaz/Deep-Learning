function [L, J] = ComputeCost(X, Y, W1, W2, b1, b2, lambda)
    [P, ~] = EvaluateClassifier(X, W1, b1, W2, b2);
    L = -sum(log(sum(Y .* P, 1))) / size(X, 2);
    
    regularization_loss = lambda * (sum(sum(W1 .^ 2)) + sum(sum(W2 .^ 2)));
    J = L + regularization_loss;
end