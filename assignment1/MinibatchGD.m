function [Wstar, bstar, train_loss, valid_loss, train_cost, valid_cost] = MiniBatchGD(X, Y, GDparams, W, b, lambda, Xval, Yval)
    n_batch = GDparams.n_batch;
    eta = GDparams.eta;
    n_epochs = GDparams.n_epochs;
    n = size(X, 2);
    
    train_loss = zeros(1, n_epochs);
    valid_loss = zeros(1, n_epochs);
    train_cost = zeros(1, n_epochs);
    valid_cost = zeros(1, n_epochs);
    
    for epoch = 1:n_epochs
        indices = randperm(n);
        X_shuffled = X(:, indices);
        Y_shuffled = Y(:, indices);
        
        for j = 1:n/n_batch
            j_start = (j-1) * n_batch + 1;
            j_end = j * n_batch;
            Xbatch = X_shuffled(:, j_start:j_end);
            Ybatch = Y_shuffled(:, j_start:j_end);
            
            P = EvaluateClassifier(Xbatch, W, b);
            [grad_W, grad_b] = ComputeGradients(Xbatch, Ybatch, P, W, lambda);
            
            W = W - eta * grad_W;
            b = b - eta * grad_b;
        end
        
        [train_loss(epoch), train_cost(epoch)] = ComputeCost(X, Y, W, b, lambda);
        [valid_loss(epoch), valid_cost(epoch)] = ComputeCost(Xval, Yval, W, b, lambda);
        
        fprintf('Epoch %d, Training Loss: %.4f\n', epoch, train_loss(epoch));
    end
    
    Wstar = W;
    bstar = b;
end