function processed_emg = emg_preprocess(raw_emg, fs)
% Fully preprocesses raw EMG data to create a smooth activity envelope
% suitable for regression tasks like joint angle prediction. (based on common practices)

% NinaPro DB8 comes already Band pass filtered (20Hz - 450 Hz)
% NinaPro DB8 is also already notch filtered at 50 Hz.

% fs is no longer used (it was used when bandpass filtering which is not
% done anymore)

% Pipeline:
% 1. Centering: Removes DC offset.
% 2. Rectification: Takes the absolute value to measure signal intensity.
% 3. Low-Pass Filtering (Smoothing): Creates the final signal envelope. (not done)
% 4. Cleanup: Ensures all values are non-negative.

% 1. Centering (Remove DC Offset)
emg_centered = raw_emg - mean(raw_emg, 1);

% 2. Rectification (Full-Wave)
emg_rectified = abs(emg_centered);

% 3. Smoothing (Envelope Detection)
%lp_cutoff = 4; % Cutoff frequency in Hz. 

% What I changed here: Reduced filter order from 4 to 2
% 2nd-order filter is more stable for low cutoff frequencies.
%lp_order = 2; 
%[b_lp, a_lp] = butter(lp_order, lp_cutoff / (fs/2), 'low');
%processed_emg_envelope = filtfilt(b_lp, a_lp, emg_rectified);


% 4. Cleanup ringing artifacts
% Ensure the final signal is non-negative.
processed_emg = max(0, emg_rectified);

end
