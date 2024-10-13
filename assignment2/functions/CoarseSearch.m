function [best_lambdas, coarse_results]  = coarseSearch(lambdas, trainNormX, trainY, trainy, validNormX, validY, validy, GDparams, input_size, output_size)
    coarse_results = struct('lambda', [], 'accuracy', []);
    input_size = 3072;
    output_size = 10;
    for i = 1:length(lambdas)
        lambda = lambdas(i);
        fprintf('Testing lambda: %f\n', lambda);

        [W1, b1, W2, b2] = Parameters(input_size, output_size); 
        [W1star, b1star, W2star, b2star, ~] = MiniBatchGD(trainNormX, trainY, trainy, GDparams, W1, W2, b1, b2, validNormX, validY, validy, lambda);
        val_acc = ComputeAccuracy(validNormX, validy, W1star, b1star, W2star, b2star);
        fprintf('Validation accuracy for lambda %f: %f\n', lambda, val_acc * 100);

        coarse_results(i).lambda = lambda;
        coarse_results(i).accuracy = val_acc;
    end

    [~, idx] = sort([coarse_results.accuracy], 'descend');
    results = coarse_results(idx);

    best_lambdas = [results(1:3).lambda];
end