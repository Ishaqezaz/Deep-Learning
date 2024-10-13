function acc = ComputeAccuracy(X, y, W, b)
    P = EvaluateClassifier(X, W, b);
    
    [~, predicted_labels] = max(P, [], 1);
    correct = sum(predicted_labels' == y); 
    acc = correct / length(y);
end