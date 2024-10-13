function [grad_W1, grad_b1, grad_W2, grad_b2] = ComputeGradients(X, Y, P, H, W1, W2, lambda)
    n = size(X, 2);

    G = P - Y;
    grad_W2 = G * H' / n + 2 * lambda * W2;
    grad_b2 = sum(G, 2) / n;
    
    G = W2' * G;
    G = G .* (H > 0);
   
    grad_W1 = G * X' / n + 2 * lambda * W1;
    grad_b1 = sum(G, 2) / n;
end