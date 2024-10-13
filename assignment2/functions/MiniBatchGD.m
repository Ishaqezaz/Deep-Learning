function [W1star, b1star, W2star, b2star, metrics] = MiniBatchGD(X, Y, y, GDparams, W1, W2, b1, b2, X_val, Y_val, y_val, lambda)
    eta_min = GDparams.eta_min;
    eta_max = GDparams.eta_max;
    step_size = GDparams.step_size;
    n_batch = GDparams.n_batch;
    cycles = GDparams.cycles;
    
    metrics = InitializeMetrics(X, Y, y, W1, b1, W2, b2, X_val, Y_val, y_val, lambda);

    for cycle = 1:cycles
        for t = 1:2 * step_size
            eta = EtaUpdate(t, cycle, step_size, eta_min, eta_max);
            [X_batch, Y_batch] = CreateBatches(X, Y, n_batch);
            [P_batch, H_batch] = EvaluateClassifier(X_batch, W1, b1, W2, b2);
            [grad_W1, grad_b1, grad_W2, grad_b2] = ComputeGradients(X_batch, Y_batch, P_batch, H_batch, W1, W2, lambda);

            [W1, b1, W2, b2] = UpdateParameters(W1, b1, W2, b2, grad_W1, grad_b1, grad_W2, grad_b2, eta);
             if mod(t, 100) == 0 || t == 800  
                if any(isnan(grad_W1(:))) || any(isnan(grad_W2(:))) || any(isnan(W1(:))) || any(isnan(W2(:)))
                    disp('NaN detected in gradients or parameters');
                end
            end

            if mod(t, step_size / 10) == 0 || t == 2 * step_size  
                metrics = LogMetrics(X, Y, y, W1, b1, W2, b2, X_val, Y_val, y_val, lambda, metrics, t + (cycle - 1) * 2 * step_size);
            end
        end
    end

    W1star = W1;
    b1star = b1;
    W2star = W2;
    b2star = b2;
end

% Prepare batches
function [X_batch, Y_batch] = CreateBatches(X, Y, n_batch)
    indices = randperm(size(X, 2));
    X_batch = X(:, indices(1:n_batch));
    Y_batch = Y(:, indices(1:n_batch));
end

% Cyclic eta
function eta = EtaUpdate(t, cycle, step_size, eta_min, eta_max)
    t_cycle = mod(t - 1, 2 * step_size) + 1;

    if t_cycle <= step_size
        eta = eta_min + (eta_max - eta_min) * t_cycle / step_size;
    else
        eta = eta_max - (eta_max - eta_min) * (t_cycle - step_size) / step_size;
    end
end

% Updating parameters
function [W1, b1, W2, b2] = UpdateParameters(W1, b1, W2, b2, grad_W1, grad_b1, grad_W2, grad_b2, eta)
    W1 = W1 - eta * grad_W1;
    b1 = b1 - eta * grad_b1;
    W2 = W2 - eta * grad_W2;
    b2 = b2 - eta * grad_b2;
end

% Logging metrics
function metrics = LogMetrics(X, Y, y, W1, b1, W2, b2, X_val, Y_val, y_val, lambda, metrics, t)
    [loss_train, cost_train] = ComputeCost(X, Y, W1, W2, b1, b2, lambda);
    acc_train = ComputeAccuracy(X, y, W1, b1, W2, b2);

    metrics.updates = [metrics.updates, t];
    metrics.loss_train = [metrics.loss_train, loss_train];
    metrics.cost_train = [metrics.cost_train, cost_train];
    metrics.acc_train = [metrics.acc_train, acc_train];
    
    if ~isempty(X_val)
        [loss_val, cost_val] = ComputeCost(X_val, Y_val, W1, W2, b1, b2, lambda);
        acc_val = ComputeAccuracy(X_val, y_val, W1, b1, W2, b2);
        metrics.loss_val = [metrics.loss_val, loss_val];
        metrics.cost_val = [metrics.cost_val, cost_val];
        metrics.acc_val = [metrics.acc_val, acc_val];
    end
end

function metrics = InitializeMetrics(X, Y, y, W1, b1, W2, b2, X_val, Y_val, y_val, lambda)
    [initial_loss_train, initial_cost_train] = ComputeCost(X, Y, W1, W2, b1, b2, lambda);
    initial_acc_train = ComputeAccuracy(X, y, W1, b1, W2, b2);

    metrics.updates = 0;
    metrics.loss_train = initial_loss_train;
    metrics.cost_train = initial_cost_train;
    metrics.acc_train = initial_acc_train;

    [initial_loss_val, initial_cost_val] = ComputeCost(X_val, Y_val, W1, W2, b1, b2, lambda);
    initial_acc_val = ComputeAccuracy(X_val, y_val, W1, b1, W2, b2);
    metrics.loss_val = initial_loss_val;
    metrics.cost_val = initial_cost_val;
    metrics.acc_val = initial_acc_val;

end