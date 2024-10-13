rng(400);
% Network input
input_size = 3072;
output_size = 10;

% Loading and preprocessing
[trainX, trainY, trainy] = loadBatch('data_batch_1.mat');
[validX, validY, validy] = loadBatch('data_batch_2.mat');
[testX, testY, testy] = loadBatch('test_batch.mat');
[trainNormX, validNormX, testNormX] = PreprocessData(trainX, validX, testX);

% Network
[W1, b1, W2, b2] = Parameters(input_size, output_size); 

% GD input
GDparams.n_batch = 100; 
GDparams.cycles = 1;
lambda = 0.01;
GDparams.eta_min = 1e-5;
GDparams.eta_max = 1e-1;
GDparams.step_size = 500;


% Mini-batch algorithm
[W1star, b1star, W2star, b2star, metrics] = MiniBatchGD(trainNormX, trainY, trainy, GDparams, W1, W2, b1, b2, validNormX, validY, validy, lambda);
total_test_acc = ComputeAccuracy(testNormX, testy, W1star, b1star, W2star, b2star);
fprintf('Total test accuracy: %.2f%%\n', total_test_acc * 100);
Graphs(metrics)

%% SCRIPT FOR FINDING THE BEST LAMBDA
rng(400)
[trainNormX, validNormX, testNormX, trainY, validY, testY, trainy, validy, testy] = PrepareData(5000);


% GD input
GDparams.n_batch = 100; 
GDparams.cycles = 2;
GDparams.eta_min = 1e-5;
GDparams.eta_max = 1e-1;
total_training_samples = size(trainNormX, 2);
GDparams.step_size = 2 * floor(total_training_samples / GDparams.n_batch);

% COARSE AND FINE
lambdas = logspace(-5, -1, 8);
[best_lambdas, coarse_results] = CoarseSearch(lambdas, trainNormX, trainY, trainy, validNormX, validY, validy, GDparams);
[best_lambda, fine_results] = FineSearch(lambdas, trainNormX, trainY, trainy, validNormX, validY, validy, GDparams);

% Best model
[trainNormX, validNormX, testNormX, trainY, validY, testY, trainy, validy, testy] = prepare_data(1000);
GDparams.n_batch = 100; 
GDparams.cycles = 3;
GDparams.eta_min = 1e-5;
GDparams.eta_max = 1e-1;
total_training_samples = size(trainNormX, 2);
GDparams.step_size = 2 * floor(total_training_samples / GDparams.n_batch);

input_size = 3072;
output_size = 10;
[W1, b1, W2, b2] = Parameters(input_size, output_size); 

[W1star, b1star, W2star, b2star, metrics] = MiniBatchGD(trainNormX, trainY, trainy, GDparams, W1, W2, b1, b2, validNormX, validY, validy, best_lambda);
test_acc = ComputeAccuracy(testNormX, testy, W1star, b1star, W2star, b2star);
fprintf('Total test accuracy: %f\n', test_acc);

Graphs(metrics);