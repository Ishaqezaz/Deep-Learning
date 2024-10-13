function [trainNormX, validNormX, testNormX] = PreprocessData(trainX, validX, testX)
    meanX = mean(trainX, 2);
    stdX = std(trainX, 0, 2);
    
    trainNormX = trainX - repmat(meanX, [1, size(trainX, 2)]);
    trainNormX = trainNormX ./ repmat(stdX, [1, size(trainNormX, 2)]);
    
    validNormX = validX - repmat(meanX, [1, size(validX, 2)]);
    validNormX = validNormX ./ repmat(stdX, [1, size(validNormX, 2)]);
    
    testNormX = testX - repmat(meanX, [1, size(testX, 2)]);
    testNormX = testNormX ./ repmat(stdX, [1, size(testNormX, 2)]);

end