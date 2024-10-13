%% Set the random seed for reproducibility
rng(400);

%% Load and preprocess data
[trainX, trainY, trainy] = loadBatch('data_batch_1.mat');
[validX, validY, validy] = loadBatch('data_batch_2.mat');
[testX, testY, testy] = loadBatch('test_batch.mat');
[trainNormX, validNormX, testNormX] = PreprocessData(trainX, validX, testX);

%% Initialize parameters
K = 10;
d = 3072;
W = randn(K, d) * 0.01;
b = randn(K, 1) * 0.01;

%% Set GD parameters
GDparams.n_batch = 100;
GDparams.eta = 0.001;
GDparams.n_epochs = 40;
lambda = 1;

%% Computing Gradient
[Wstar, bstar, train_loss, valid_loss, train_cost, valid_cost] = MiniBatchGD(trainNormX, trainY, GDparams, W, b, lambda, validNormX, validY);

%% Graphings
% Visualize the weight matrix
for i = 1:K
    im = reshape(Wstar(i, :), 32, 32, 3);
    im = (im - min(im(:))) / (max(im(:)) - min(im(:)));
    s_im{i} = permute(im, [2, 1, 3]);
end
montage(s_im, 'Size', [2, 5]);

% Plotting training and validation loss
figure;
plot(train_loss, 'y-', 'LineWidth', 2);
hold on;
plot(valid_loss, 'r-', 'LineWidth', 2);
hold off;

title('Training and Validation Loss Over Epochs', 'FontSize', 18);
xlabel('Epoch', 'FontSize', 16);
ylabel('Loss', 'FontSize', 16);
legend('Training Loss', 'Validation Loss', 'FontSize', 14);
grid on;

% Plotting training and validation cost
figure;
plot(train_cost, 'y-', 'LineWidth', 2);
hold on;
plot(valid_cost, 'r-', 'LineWidth', 2);
title('Training and Validation Cost Over Epochs', 'FontSize', 18);
xlabel('Epoch', 'FontSize', 16);
ylabel('Cost', 'FontSize', 16);
legend('Training Cost', 'Validation Cost', 'FontSize', 14);
grid on;
hold off;

%% Total Accuracy of the model
total_test_acc = ComputeAccuracy(testNormX, testy, Wstar, bstar);
fprintf('Total test accuracy: %.2f%%\n', total_test_acc * 100);