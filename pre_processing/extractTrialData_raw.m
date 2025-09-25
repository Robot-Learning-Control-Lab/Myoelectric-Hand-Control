function [X_windows,Y_windows,varargout] = extractTrialData_raw(...
    rerepetition, restimulus, targetMovement, T, ...
    emg_raw, glove_raw, ... % Pass in RAW emg and NORMALIZED glove
    emg_channels, glove_channels, win_stride, fs, varargin) % Add scaling_params as varargin for val and tst data

    % Find out how many repetitions exist for the target movement
    num_repetitions = max(rerepetition(restimulus == targetMovement));
    fprintf('Found %d repetitions for movement %d.\n', num_repetitions, targetMovement);
    
    % Initialize cell arrays
    X_windows = {};
    Y_windows = {};
    
    % Loop through each repetition for the target movement
    for rep = 1:num_repetitions
        % Find the indices for this specific trial
        trial_indices = find(restimulus == targetMovement & rerepetition == rep);
        
        if length(trial_indices) < T
            fprintf('    Skipping repetition #%d, not enough data.\n', rep);
            continue;
        end
        
        % Extract the RAW EMG and DoFs for this trial
        emg_trial_raw = emg_raw(trial_indices, emg_channels);
        glove_trial_raw = glove_raw(trial_indices, glove_channels);

        % Preprocess only this trial's EMG and DoF data
        emg_trial_processed = emg_preprocess(emg_trial_raw, fs);

        if nargin == 10
            [glove_trial_normalized, scaling_params] = glove_preprocess(glove_trial_raw);

            varargout{1} = scaling_params; % Return the parameters

        elseif nargin == 11
            scaling_params = varargin{1};
            glove_trial_normalized = glove_preprocess(glove_trial_raw, scaling_params);

        end
        
        % Apply the sliding window to the PROCESSED trial data
        start_idx = 1;
        while (start_idx + T - 1) <= size(emg_trial_processed, 1)
            end_idx = start_idx + T - 1;
            
            % Extract one window of data
            emg_window = emg_trial_processed(start_idx:end_idx, :)';
            glove_point = glove_trial_normalized(end_idx, :)';
            
            % Add the new windows to our master lists
            X_windows{end+1} = single(emg_window);
            Y_windows{end+1} = single(glove_point);
            
            % Move the window start index forward
            start_idx = start_idx + win_stride;
        end
    end
end