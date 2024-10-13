function [W1, b1, W2, b2] = Parameters(input_dimension, output_dimension)
    dev1 = 1/sqrt(input_dimension);
    hidden_neurons = 50;
    dev2 = 1/sqrt(hidden_neurons);

    W1 = randn(hidden_neurons, input_dimension).*dev1;
    b1 = zeros(hidden_neurons, 1);
    
    W2 = randn(output_dimension, hidden_neurons).*dev2;
    b2 = zeros(output_dimension, 1);
end