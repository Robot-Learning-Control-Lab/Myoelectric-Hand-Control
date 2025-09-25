function [gradients, loss] = modelGradients_seq2vec(net, dlX, dlY_true)
% This function calculates the gradients and loss for a model.
% The loss function used is Mean Squared Error (MSE).
% Training is done in batches of size 16 (batchSize) so the dlY_pred gives 16 2x1 vectors
% for a batch of 16 emg inputs

    % Get model prediction by passing the input data through the network.
    % dlX is size: 16  x 256   x  1  x  16
    dlY_pred = forward(net, dlX);

    % Calculate the MSE loss between the prediction and the true values.
    %loss = mse(dlY_pred, dlY_true, 'DataFormat', 'CB');
    loss = mse(dlY_pred, dlY_true);
    
    % Compute the gradients of the loss with respect to the network's
    % learnable parameters (weights and biases).
    gradients = dlgradient(loss, net.Learnables);

%% Add Physics-Informed loss (PIL) here later (Geometric Variable Strain Lagrangian EoM)
% feed accelereation, gyroscope and glove kinematics to function for PIL


end