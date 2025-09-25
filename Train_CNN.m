%% Train_SeqToVec_2D_CNN (MANUAL BATCHING)
clear; close all; clc;
rng('default');

%% 1. Load and Prepare Data
disp('Loading and Preparing Data');
s = 1; m = 3; % Subject (s) and Target movement (m)
base_folder = 'dlData';
fileName = sprintf('train_val_tst_S%d_M%d.mat', s, m);
full_path = fullfile(base_folder, fileName);
data = load(full_path);

% Extract data
XTrainCell = data.Xtrain; YTrainCell = data.Ytrain;
XValCell = data.Xval;     YValCell = data.Yval;
XTstCell = data.Xtst;     YTstCell = data.Ytst;

% Data parameters
C = size(XTrainCell{1}, 1); F = size(YTrainCell{1}, 1); T = size(XTrainCell{1}, 2);
numTrainSamples = numel(XTrainCell);
numValSamples = numel(XValCell);

% Prepare Y data (Output Vectors)
YTrain_vec = cellfun(@(c) c(:,end), YTrainCell, 'UniformOutput', false);
YVal_vec   = cellfun(@(c) c(:,end), YValCell,   'UniformOutput', false);
YTst_vec   = cellfun(@(c) c(:,end), YTstCell,   'UniformOutput', false);

% Normalize input data (z-score)
disp('Normalizing Input Data (X)');
XTrainMat_for_stats = cat(2, XTrainCell{:});
mu = mean(XTrainMat_for_stats, 2);
sigma = std(XTrainMat_for_stats, 0, 2);
XTrainCell = cellfun(@(x) (x - mu) ./ sigma, XTrainCell, 'UniformOutput', false);
XValCell   = cellfun(@(x) (x - mu) ./ sigma, XValCell,   'UniformOutput', false);
XTstCell   = cellfun(@(x) (x - mu) ./ sigma, XTstCell,   'UniformOutput', false);

% Create final matrices for training
XTrain4D = cat(4, XTrainCell{:});
XTrain4D = reshape(XTrain4D, C, T, 1, []); % Shape: [16, 256, 1, N]
YTrainMat = cat(2, YTrain_vec{:}); % Shape: [2, N]

XVal4D = cat(4, XValCell{:});
XVal4D = reshape(XVal4D, C, T, 1, []);
YValMat = cat(2, YVal_vec{:});

%% 2. Define 2D CNN model architecture
disp('Defining Full 2D CNN Architecture');
inputSize = [C, T, 1];
layers = [
    imageInputLayer(inputSize, 'Name', 'input', 'Normalization', 'none')
    convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    dropoutLayer(0.3, 'Name', 'drop1')
    convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    dropoutLayer(0.3, 'Name', 'drop2')
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv3')
    batchNormalizationLayer('Name', 'bn3')
    reluLayer('Name', 'relu3')
    dropoutLayer(0.3, 'Name', 'drop3')
    maxPooling2dLayer(3, 'Padding', 'same', 'Stride', 2, 'Name', 'maxpool')
    flattenLayer('Name', 'flatten')
    fullyConnectedLayer(128, 'Name', 'fc1')
    batchNormalizationLayer('Name', 'bn_fc1')
    reluLayer('Name', 'relu_fc1')
    dropoutLayer(0.3, 'Name', 'drop_fc1')
    fullyConnectedLayer(128, 'Name', 'fc2')
    batchNormalizationLayer('Name', 'bn_fc2')
    reluLayer('Name', 'relu_fc2')
    dropoutLayer(0.3, 'Name', 'drop_fc2')
    fullyConnectedLayer(F, 'Name', 'output')
];
net = dlnetwork(layerGraph(layers));

% %% 2. Deeper and Wider 2D CNN Model Architecture
% disp('Defining Deeper and Wider 2D CNN Architecture with L2 Regularization');
% 
% inputSize = [C, T, 1];
% l2_reg = 1e-3; % L2 Regularization factor
% drop_rate = 0.5; % Increased dropout rate
% 
% layers = [
%     imageInputLayer(inputSize, 'Name', 'input', 'Normalization', 'none')
% 
%     % --- Wider First Conv Block ---
%     convolution2dLayer(3, 256, 'Padding', 'same', 'Name', 'conv1', 'WeightL2Factor', l2_reg)
%     batchNormalizationLayer('Name', 'bn1')
%     reluLayer('Name', 'relu1')
%     dropoutLayer(drop_rate, 'Name', 'drop1')
% 
%     % --- Wider Second Conv Block ---
%     convolution2dLayer(3, 256, 'Padding', 'same', 'Name', 'conv2', 'WeightL2Factor', l2_reg)
%     batchNormalizationLayer('Name', 'bn2')
%     reluLayer('Name', 'relu2')
%     dropoutLayer(drop_rate, 'Name', 'drop2')
% 
%     % --- New Deeper Third Conv Block ---
%     convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv3', 'WeightL2Factor', l2_reg)
%     batchNormalizationLayer('Name', 'bn3')
%     reluLayer('Name', 'relu3')
%     dropoutLayer(drop_rate, 'Name', 'drop3')
% 
%     maxPooling2dLayer(3, 'Padding', 'same', 'Stride', 2, 'Name', 'maxpool1')
% 
%     % --- Original Third Conv Block is now the Fourth ---
%     convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv4', 'WeightL2Factor', l2_reg)
%     batchNormalizationLayer('Name', 'bn4')
%     reluLayer('Name', 'relu4')
%     dropoutLayer(drop_rate, 'Name', 'drop4')
% 
%     maxPooling2dLayer(3, 'Padding', 'same', 'Stride', 2, 'Name', 'maxpool2')
% 
%     flattenLayer('Name', 'flatten')
% 
%     % --- Wider First FC Block ---
%     fullyConnectedLayer(256, 'Name', 'fc1', 'WeightL2Factor', l2_reg)
%     batchNormalizationLayer('Name', 'bn_fc1')
%     reluLayer('Name', 'relu_fc1')
%     dropoutLayer(drop_rate, 'Name', 'drop_fc1')
% 
%     % --- Wider Second FC Block ---
%     fullyConnectedLayer(256, 'Name', 'fc2', 'WeightL2Factor', l2_reg)
%     batchNormalizationLayer('Name', 'bn_fc2')
%     reluLayer('Name', 'relu_fc2')
%     dropoutLayer(drop_rate, 'Name', 'drop_fc2')
% 
%     % --- New Deeper Third FC Block ---
%     fullyConnectedLayer(128, 'Name', 'fc3', 'WeightL2Factor', l2_reg)
%     batchNormalizationLayer('Name', 'bn_fc3')
%     reluLayer('Name', 'relu_fc3')
%     dropoutLayer(drop_rate, 'Name', 'drop_fc3')
% 
%     % --- Final Output Layer ---
%     fullyConnectedLayer(F, 'Name', 'output')
% ];
% 
% net = dlnetwork(layerGraph(layers));


%% 3. Training loop with manual batching
disp('Starting Training');
% Hyperparameters
epochs = 300; batchSize = 16; initialLearnRate = 0.001;
decayRate = 0.9; decaySteps = 300; patience = 20;

% Initialize training state
iteration = 0; bestValLoss = inf; epochsWithoutImprovement = 0; bestNet = net; 
trailingAvg = []; trailingAvgSq = [];

% Training and validation progress tracking plot
disp('Initializing Live Training Plot');
figure;
grid on;
hold on; 
xlabel("Epoch");
ylabel("Loss (MSE)");
title("Live Training Progress");

% Create animated lines for training and validation loss
trainLossPlot = animatedline('Color', [0, 0.4470, 0.7410], 'LineWidth', 1.5);
valLossPlot = animatedline('Color', [0.8500, 0.3250, 0.0980], 'LineWidth', 1.5);
legend('Avg. Training Loss', 'Validation Loss');

% Start timer for training duration
totalTrainingStart = tic;

for epoch = 1:epochs
    % Manual shuffling of training points 
    idx = randperm(numTrainSamples);
    XTrain4D = XTrain4D(:,:,:,idx);
    YTrainMat = YTrainMat(:,idx);

    % Variables to accumulate training loss for the epoch
    trainLossAccum = 0; 
    numTrainBatches = 0;
    
    for i = 1:batchSize:numTrainSamples
        iteration = iteration + 1;
        
        % Manual batching
        batchIdx = i:min(i+batchSize-1, numTrainSamples);
        dlX = dlarray(XTrain4D(:,:,:,batchIdx), 'SSCB');
        dlY = dlarray(YTrainMat(:,batchIdx), 'CB');
        
        % Learning rate decay (dynamic learning rate)
        learnRate = initialLearnRate * decayRate^(iteration / decaySteps);
        
        % Evaluate gradients and loss 
        [grads, loss] = dlfeval(@modelGradients_seq2vec, net, dlX, dlY);

        % Update network 
        [net, trailingAvg, trailingAvgSq] = adamupdate(net, grads, ...
            trailingAvg, trailingAvgSq, iteration, learnRate);


         % Accumulate the training loss for this batch
        trainLossAccum = trainLossAccum + extractdata(loss);
        numTrainBatches = numTrainBatches + 1;
    end

     % Calculate the average training loss for the epoch
    avgTrainLoss = trainLossAccum / numTrainBatches;
    
    % Validation 
    dlX_val = dlarray(XVal4D, 'SSCB');
    dlY_val = dlarray(YValMat, 'CB');
    dlY_pred_val = predict(net, dlX_val);

    valLoss = mse(dlY_pred_val, dlY_val);
    valLoss = extractdata(valLoss);
    
    fprintf('Epoch %d | Validation Loss: %f\n', epoch, valLoss);

    % Update tracking live plot
    addpoints(trainLossPlot, epoch, avgTrainLoss);
    addpoints(valLossPlot, epoch, valLoss);
    drawnow; 
    
    % Early stopping logic 
    % (if the val. loss doesn't improve for "patience" epochs stop the training)

    if valLoss < bestValLoss
        bestValLoss = valLoss;
        bestNet = net;

        epochsWithoutImprovement = 0;
    else
        epochsWithoutImprovement = epochsWithoutImprovement + 1;
    end

    
    if epochsWithoutImprovement >= patience
        fprintf('Early stopping triggered after %d epochs.\n', epoch);
        break;
    end
end

% the final network (weights) is the network with the min Val. loss 
net = bestNet; 

% Stop timer for training duration
totalTrainingTimeSec = toc(totalTrainingStart);
totalTrainingTimeMin = totalTrainingTimeSec / 60;

disp('--- Training Finished ---');
fprintf('Total Training Time: %.2f seconds (%.2f minutes)\n', totalTrainingTimeSec, totalTrainingTimeMin);

%% 5. Evaluate and Visualize (Training Data Scatter plots)
disp('Evaluating on Training Data');

XTrain_with_channel = add_channel_dim_to_cell(XTrainCell);
XTrain4D = cat(4, XTrain_with_channel{:});
dlXTrain = dlarray(XTrain4D, 'SSCB');

dlYPred = predict(net, dlXTrain);

y_pred_mat = extractdata(dlYPred)';
y_true_mat = cat(2, YTrain_vec{:})';

% Scatter Plot 
figure;
sgtitle('2D CNN: Test Predictions vs. True Angles', 'FontSize', 14);
subplot(1, 2, 1);
scatter(y_true_mat(:,1), y_pred_mat(:,1), 'filled');
xlabel('True Angle'); ylabel('Predicted Angle'); title('DoF 1'); grid on; refline(1,0);
subplot(1, 2, 2);
scatter(y_true_mat(:,2), y_pred_mat(:,2), 'r', 'filled');
xlabel('True Angle'); ylabel('Predicted Angle'); title('DoF 2'); grid on; refline(1,0);


disp('Evaluating on Training Data');

%% 5. Evaluate and Visualize (Test Data Scatter plots)
disp('Evaluating on Test Data');

XTst_with_channel = add_channel_dim_to_cell(XTstCell);
XTst4D = cat(4, XTst_with_channel{:});
dlXTst = dlarray(XTst4D, 'SSCB');

dlYPred = predict(net, dlXTst);

y_pred_mat = extractdata(dlYPred)';
y_true_mat = cat(2, YTst_vec{:})';

% Scatter Plot
figure;
sgtitle('2D CNN: Test Predictions vs. True Angles', 'FontSize', 14);
subplot(1, 2, 1);
scatter(y_true_mat(:,1), y_pred_mat(:,1), 'filled');
xlabel('True Angle'); ylabel('Predicted Angle'); title('DoF 1'); grid on; refline(1,0);
subplot(1, 2, 2);
scatter(y_true_mat(:,2), y_pred_mat(:,2), 'r', 'filled');
xlabel('True Angle'); ylabel('Predicted Angle'); title('DoF 2'); grid on; refline(1,0);

%% Plot Normalized DoFs

figure;
plot(y_pred_mat(:,1))
hold on;
plot(y_true_mat(:,1))
xlabel('Time')
ylabel('DoF 1 (Normalized)')


figure;
plot(y_pred_mat(:,2))
hold on;
plot(y_true_mat(:,2))
xlabel('Time')
ylabel('DoF 2 (Normalized)')
