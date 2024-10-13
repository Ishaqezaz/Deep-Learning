function [trainNormX, validNormX, testNormX, trainY, validY, testY, trainy, validy, testy] = prepare_data(validationSize)
    [X_1, YY_1, y_1] = loadBatch('data/data_batch_1.mat');
    [X_2, YY_2, y_2] = loadBatch('data/data_batch_2.mat');
    [X_3, YY_3, y_3] = loadBatch('data/data_batch_3');
    [X_4, YY_4, y_4] = loadBatch('data/data_batch_4.mat');
    [X_5, YY_5, y_5] = loadBatch('data/data_batch_5.mat');
    [testX, testY, testy] = loadBatch('data/test_batch.mat');

    X_all = [X_1, X_2, X_3, X_4, X_5];
    YY_all = [YY_1, YY_2, YY_3, YY_4, YY_5];
    y_all = [y_1, y_2, y_3, y_4, y_5];

    totalSamples = size(X_all, 2);
    idxStart = totalSamples - validationSize + 1;
    idxEnd = totalSamples;

    validX = X_all(:, idxStart:idxEnd);
    validY = YY_all(:, idxStart:idxEnd);
    validy = y_all(idxStart:idxEnd);

    trainX = X_all(:, 1:idxStart-1);
    trainY = YY_all(:, 1:idxStart-1);
    trainy = y_all(1:idxStart-1);

    [trainNormX, validNormX, testNormX] = PreprocessData(trainX, validX, testX);
end