function A = ComputeAccuracy(X, y, W1, b1, W2, b2)
    P = EvaluateClassifier(X, W1, b1, W2, b2);
    [~, predicted_labels] = max(P, [], 1); 
    if size(y, 1) > 1
        y = y';
    end
    correct_predictions = sum(predicted_labels == y);
    A = correct_predictions / length(y);
end