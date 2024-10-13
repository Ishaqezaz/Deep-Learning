function P = EvaluateClassifier(X, W, b)
    s = W*X+b;
    exps = exp(s - max(s, [], 1));
    P = exps ./ sum(exps, 1);
end