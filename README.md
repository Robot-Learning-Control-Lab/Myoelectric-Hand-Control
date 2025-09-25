# Continuous Hand Movement Prediction from sEMG using a CNN

This repository contains the MATLAB implementation of a Neural Network (CNN) designed to decode continuous hand movements from muscle activity. The model learns the complex mapping from high-density surface electromyography (sEMG) signals to 2-Degree-of-Freedom (DoF) Index finger kinematics. The project is trained and evaluated on the NinaPro database (DB8).

## Key Features

* **Sequence-to-Vector Regression:** Predicts the DoF angles at a specific time step based on a 256-sample window of historical EMG data.
* **Live Training Visualization:** Plots training and validation loss in real-time for immediate feedback on model performance.
* **Best Model Saving:** Implements early stopping logic to automatically save the model with the lowest validation loss.

## Data

This model is designed to work with data from the **NinaPro (Non-Invasive Adaptive Prosthetics)** database 8. 
You can access the dataset and more info at <[DB8](https://ninapro.hevs.ch/instructions/DB8.html)>.

Before running the script, your data must be pre-processed and saved into a `.mat` file with the following structure:

* `Xtrain`, `Xval`, `Xtst`: Cell arrays containing the EMG data windows.
* `Ytrain`, `Yval`, `Yst`: Cell arrays containing the corresponding DoF/glove data.

This is already done for Subject 1 Movements 1 to 5.

The main training script expects these `.mat` files to be in a subfolder named `dlData`.
You need to download the `dlData` folder to run and train the Network: 

## Prerequisites

* **MATLAB** (tested on R2025a)
* **Deep Learning Toolbox**

## How to Run

1.  **Download Data:** Download the `dlData` folder and place it in the project's root directory. Link to `dlData`: <[dlData folder](https://drive.google.com/drive/folders/11hzMvXQ2gO-lptywPy4m13W_k0VEyGiD?usp=sharing)>
2.  **Run Training:** Execute the script in MATLAB. The training process will begin, displaying the live loss plot and printing progress to the command window.

## Preprocessing Pipeline

### Data Extraction and Segmentation 
1.  **Trial Extraction:** The  `extractTrialData_raw()` function first identifies and isolates the continuous data corresponding to each repetition (or "trial") of a specific target movement. It uses the `restimulus` and `rerepetition` vectors from the database to find the start and end indices for each trial.
2.  **Data Pre-processing:** Once a trial is isolated the corresponding input and output data is pre-processed using `emg_preprocess()` and `glove_preprocess()` respectively (see below sections). 
3.  **Sliding Window Segmentation:** After trial data is preprocessed, a sliding window is applied to segment the data.
    * A window of a fixed length (`T` = 256 samples) is moved across the trial's time-series data.
    * The window is advanced by a set number of samples (`win_stride`) computed based on 60% overlap.
    * For each position of the window, an input-output pair is created:
        * **Input (X):** The full `16 x 256` block of EMG data within the window.
        * **Output (Y):** The single `2 x 1` DoF data point from the **very last sample** of the corresponding window.

### Input (sEMG) Pre-processsing

1.  **Base Filtering (NinaPro DB8):** The script assumes the raw data has already been notch-filtered (50 Hz) and bandpass-filtered (20-450 Hz), as specified by the NinaPro dataset documentation.
2.  **DC Centering & Rectification:** The loaded signal is first centered around zero and then rectified (`abs()`) to convert the oscillating EMG into a positive-only signal representing muscle activation intensity.
3.  **Z-score Standardization:** A final standardization step is applied *within the training script*. This centers the rectified signal to have a mean of 0 and a standard deviation of 1, which is critical for stable and effective network training.

### Output (DoF / Glove Data)  Pre-processsing

1.  **Min-Max Normalization:** The output DoF data is normalized so that it's within a range of 0 to 1.

### Running Preprocessing files
1. To run the pre-processing files, the original NinaPro DB8 files need to be downloaded.


## Model Architecture

The network is a 2D CNN designed for feature extraction from spatio-temporal data. The `analyzeNetwork(net)` function is included in the script to visualize this architecture.

* **Input Layer:** Accepts a `16x256x1` "image" of EMG data.
* **Convolutional Blocks:** Three convolutional blocks with `[128, 128, 64]` filters are used to progressively learn more complex patterns. Each block consists of:
    * `convolution2dLayer`
    * `batchNormalizationLayer`
    * `reluLayer`
    * `dropoutLayer` (0.3 rate)
* **Max Pooling Layer:** Used to downsample the feature maps.
* **Fully Connected Blocks:** Two fully connected blocks with `[128, 128]` neurons act as the final decision-making layers, learning the global relationships between all the extracted features.
* **Output Layer:** A final `fullyConnectedLayer` with 2 neurons outputs the predictions for DoF 1 and DoF 2.
