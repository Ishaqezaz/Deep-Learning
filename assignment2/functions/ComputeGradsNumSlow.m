function [grad_W1, grad_b1, grad_W2, grad_b2] = ComputeGradsNumSlow(X, Y, W1, W2, b1, b2, lambda, h)
    grad_W1 = zeros(size(W1));
    grad_b1 = zeros(size(b1));
    grad_W2 = zeros(size(W2));
    grad_b2 = zeros(size(b2));
    
    for i = 1:size(W1, 1)
        for j = 1:size(W1, 2)
            W1_try = W1;
            W1_try(i, j) = W1_try(i, j) - h;
            c1 = ComputeCost(X, Y, W1_try, W2, b1, b2, lambda);
            
            W1_try(i, j) = W1_try(i, j) + 2 * h;
            c2 = ComputeCost(X, Y, W1_try, W2, b1, b2, lambda);
            
            grad_W1(i, j) = (c2 - c1) / (2 * h);
        end
    end

    for i = 1:numel(b1)
        b1_try = b1;
        b1_try(i) = b1_try(i) - h;
        c1 = ComputeCost(X, Y, W1, W2, b1_try, b2, lambda);
        
        b1_try(i) = b1_try(i) + 2 * h;
        c2 = ComputeCost(X, Y, W1, W2, b1_try, b2, lambda);
        
        grad_b1(i) = (c2 - c1) / (2 * h);
    end

    for i = 1:size(W2, 1)
        for j = 1:size(W2, 2)
            W2_try = W2;
            W2_try(i, j) = W2_try(i, j) - h;
            c1 = ComputeCost(X, Y, W1, W2_try, b1, b2, lambda);
            
            W2_try(i, j) = W2_try(i, j) + 2 * h;
            c2 = ComputeCost(X, Y, W1, W2_try, b1, b2, lambda);
            
            grad_W2(i, j) = (c2 - c1) / (2 * h);
        end
    end

    for i = 1:numel(b2)
        b2_try = b2;
        b2_try(i) = b2_try(i) - h;
        c1 = ComputeCost(X, Y, W1, W2, b1, b2_try, lambda);
        
        b2_try(i) = b2_try(i) + 2 * h;
        c2 = ComputeCost(X, Y, W1, W2, b1, b2_try, lambda);
        
        grad_b2(i) = (c2 - c1) / (2 * h);
    end
end