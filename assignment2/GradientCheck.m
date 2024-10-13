rng(400);
addpath("functions")
addpath("data")
% Network input
input_size = 3072;
output_size = 10;

% Loading and preprocessing
[trainX, trainY, trainy] = LoadBatch('data_batch_1.mat');
[validX, validY, validy] = LoadBatch('data_batch_2.mat');
[testX, testY, testy] = LoadBatch('test_batch.mat');
[trainNormX, validNormX, testNormX] = PreprocessData(trainX, validX, testX);

% PARAMETERS FOR COMPARING GRADE COMPUTATION
[W1, b1, W2, b2] = Parameters(input_size, output_size); 
X_small = trainNormX(:, 1:20);
Y_small = trainY(:, 1:20);  
W1_small = W1;
W2_small = W2;
lambda = 0;
[P, H] = EvaluateClassifier(X_small, W1_small, b1, W2_small, b2);

% GRADIENT CHECK NUMERICAL VS ANALYTICAL
% Analytucal
[agradW1, agradb1, agradW2, agradb2] = ComputeGradients(X_small, Y_small, P, H, W1_small, W2_small, lambda);
% Numerical
h = 1e-5;
[ngradW1, ngradb1, ngradW2, ngradb2] = ComputeGradsNumSlow(X_small, Y_small, W1_small, W2_small, b1, b2, lambda, h);

% Compute relative errors
eps = 1e-7;
relative_error_W1 = norm(agradW1 - ngradW1) / max(eps, norm(agradW1) + norm(ngradW1));
relative_error_b1 = norm(agradb1 - ngradb1) / max(eps, norm(agradb1) + norm(ngradb1));
relative_error_W2 = norm(agradW2 - ngradW2) / max(eps, norm(agradW2) + norm(ngradW2));
relative_error_b2 = norm(agradb2 - ngradb2) / max(eps, norm(agradb2) + norm(ngradb2));


% Display the relative errors a
fprintf('Relative error for W1: %g\n', relative_error_W1);
fprintf('Relative error for b1: %g\n', relative_error_b1);
fprintf('Relative error for W2: %g\n', relative_error_W2);
fprintf('Relative error for b2: %g\n', relative_error_b2);