function [best_lambda, fine_results] = fineSearch(best_lambdas, trainNormX, trainY, trainy, validNormX, validY, validy, GDparams)
    fine_lambdas = logspace(log10(min(best_lambdas)), log10(max(best_lambdas)), 10);
    fine_results = struct('lambda', [], 'accuracy', []);
    input_size = 3072;
    output_size = 10;

    for i = 1:length(fine_lambdas)
        lambda = fine_lambdas(i);
        fprintf('Testing lambda: %f\n', lambda);

        [W1, b1, W2, b2] = Parameters(input_size, output_size); 
        [W1star, b1star, W2star, b2star, ~] = MiniBatchGD(trainNormX, trainY, trainy, GDparams, W1, W2, b1, b2, validNormX, validY, validy, lambda);
        val_acc = ComputeAccuracy(validNormX, validy, W1star, b1star, W2star, b2star);
        fprintf('Validation accuracy for lambda %f: %f\n', lambda, val_acc * 100);

        fine_results(i).lambda = lambda;
        fine_results(i).accuracy = val_acc;
    end

    [~, idx] = max([fine_results.accuracy]);
    best_lambda = fine_results(idx).lambda;
end