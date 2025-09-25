function [glove_norm, varargout] = glove_preprocess(glove_raw, varargin)
    if nargin == 1
        % Training mode 
        % calculate min/max values from training data:
        
        min_vals = min(glove_raw, [], 1);
        max_vals = max(glove_raw, [], 1);
        
        % Calculate the range, handling channels that might be constant
        range_vals = max_vals - min_vals;
        range_vals(range_vals == 0) = 1; % Avoid division by zero
        
        % Normalize the data
        glove_norm = (glove_raw - min_vals) ./ range_vals;
        
        % scaling parameters struct outputs
        scaling_params.min = min_vals;
        scaling_params.max = max_vals;
        varargout{1} = scaling_params; 

        % For validation or test data scaling parameters are provided as input:
    elseif nargin == 2
        scaling_params = varargin{1};
        min_vals = scaling_params.min;
        max_vals = scaling_params.max;
        
        % Calculate the range from the training set parameters
        range_vals = max_vals - min_vals;
        range_vals(range_vals == 0) = 1;
        
        % Normalize new data using precomputed parameters
        glove_norm = (glove_raw - min_vals) ./ range_vals;
        
        % Clip data to the [0, 1] range. This handles cases where
        % validation/test data might fall slightly outside the training range.
        glove_norm = max(0, min(1, glove_norm));
        
    else
        error('Invalid number of input arguments. Use 1 for training or 2 for testing.');
    end
end
