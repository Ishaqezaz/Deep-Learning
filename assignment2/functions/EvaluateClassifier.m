function [P, H] = EvaluateClassifier(X, W1, b1, W2, b2)
    S1 = W1 * X + b1;
  
    H = max(0, S1);  
    
    S = W2 * H + b2;
   
    exp_S = exp(S - max(S, [], 1));
    P = exp_S ./ sum(exp_S, 1);
end