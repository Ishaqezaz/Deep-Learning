function [loss, cost] = ComputeCost(X, Y, W, b, lambda)
    P = EvaluateClassifier(X, W, b);

    probs = sum(Y .* P, 1);
    cross_entropy_loss = -sum(log(probs + eps)) / size(X, 2);
    regularization_term = lambda * sum(W(:) .^ 2);
    
    cost = cross_entropy_loss + regularization_term;
    loss = cross_entropy_loss;
end