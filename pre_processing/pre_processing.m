%% NinaPro DB8 Data Windowing Script for target movement (e.g., 3):
clear; clc;

for s = 1:1 % loop through s subjects (1 for now)
    %% Load subject file: File and Data Parameters

    %base_folder = 'subject_data';
    base_folder = '/Users/ikhlas_mac/Downloads';

    % file name
    fileName1 = sprintf('S%d_E1_A1.mat', s); % training set
    fileName2 = sprintf('S%d_E1_A2.mat', s); % validation set
    fileName3 = sprintf('S%d_E1_A3.mat', s); % testing set

    % path
    full_path_train = fullfile(base_folder, fileName1);
    full_path_val   = fullfile(base_folder, fileName2);
    full_path_tst   = fullfile(base_folder, fileName3);

    % Data shape parameters: (constant)
    emg_channels = 1:16;       % Columns to use for EMG

    % from DB8 
    % Windowing Parameters
    fs = 2000;                 % Sampling frequency (Hz)
    win_duration = 0.128;      % Window duration in seconds (128 ms)
    overlap_percent = 0.6;     % 60% overlap

    T = round(win_duration * fs); % Window size in samples (T)
    win_stride = round(T * (1 - overlap_percent)); % slide window by (samples)


    %% Load and prepare subject data
    fprintf('Loading S%d data from files\n',s);

    % Load and extract training data:
    traindata = load(full_path_train);

    restimulus_tr = traindata.restimulus;
    rerepetition_tr = traindata.rerepetition;
    emg_tr = traindata.emg;
    glove_tr = traindata.glove;


    % Load and extract validation data:
    valdata = load(full_path_val);

    restimulus_val = valdata.restimulus;
    rerepetition_val = valdata.rerepetition;
    emg_val = valdata.emg;
    glove_val = valdata.glove;


    % Load and extract test data:
    testdata = load(full_path_tst);

    restimulus_tst = testdata.restimulus;
    rerepetition_tst = testdata.rerepetition;
    emg_tst = testdata.emg;
    glove_tst = testdata.glove;


    fprintf('done\n');


    %%  Preprocessing Glove signals 

    % Min-Max normalization
    % [glove_tr, scaling_params] = glove_preprocess(glove_tr);
    % glove_val = glove_preprocess(glove_val, scaling_params);
    % glove_tst = glove_preprocess(glove_tst, scaling_params);
    % 
    % % Z-score normalization
    % [glove_tr, scaling_params] = glove_preprocess_standardize(glove_tr);
    % glove_val = glove_preprocess_standardize(glove_val, scaling_params);
    % glove_tst = glove_preprocess_standardize(glove_tst, scaling_params);

    %% Loop through Movements
    for targetMovement = 1:5

        % specify relevant glove channels (DoFs) for each movement
        switch targetMovement
            case 1 % Thumb Flex/Ext
                glove_channels = [2,3];
            case 2 % Thumb Abd/Add
                glove_channels = [1,2];
            case 3 % Index Flex/Ext
                glove_channels = [5,6];
            case 4 % Middle Flex/Ext
                glove_channels = [7,8];
            case 5 % Ring/Little Flex/Ext
                glove_channels = [10, 11, 13, 14];
        end

        %% Extract and save trials repetiton windows:

        % TRAINING DATA
        [Xtrain,Ytrain,scaling_params] = extractTrialData_raw(rerepetition_tr,restimulus_tr,...
            targetMovement,T,emg_tr,glove_tr,emg_channels,glove_channels,...
            win_stride,fs);

        % VALIDATION DATA
        [Xval,Yval] = extractTrialData_raw(rerepetition_val,restimulus_val,...
            targetMovement,T,emg_val,glove_val,emg_channels,glove_channels,...
            win_stride,fs,scaling_params);

        % TEST DATA
        [Xtst,Ytst] = extractTrialData_raw(rerepetition_tst,restimulus_tst,...
            targetMovement,T,emg_tst,glove_tst,emg_channels,glove_channels,...
            win_stride,fs,scaling_params);


        %% Save extracted data.
        output_folder = 'dlData';

        save_file_name = sprintf('train_val_tst_S%d_M%d.mat', s, targetMovement);
        save_path = fullfile(output_folder, save_file_name);

        save(save_path, 'Xtrain', 'Ytrain', 'Xval', 'Yval', 'Xtst', 'Ytst');

        fprintf('completed preprocessing for subject %d movement %d\n',s,targetMovement);

    end

end