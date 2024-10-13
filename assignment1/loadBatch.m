function [X, Y, y] = loadBatch(filename)
    A = load(filename);
    
    X = double(A.data) / 255.0;
    y = double(A.labels) + 1;
    
    K = 10;
    n = length(y);
    
    Y = zeros(K, n);
    for i = 1:n
        Y(y(i), i) = 1;
    end

    X = X';
   
end