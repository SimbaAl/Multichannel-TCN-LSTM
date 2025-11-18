clc;
clearvars;
close all;

% Set random seed for reproducibility
rng(42); 

% Define total number of samples
TOTAL_SAMPLES = 4000; % Total dataset size
NUM_TRAIN = 2000;      % Number of training samples
NUM_TEST = 2000;        % Number of testing samples

% Ensure the sum is correct
assert(NUM_TRAIN + NUM_TEST == TOTAL_SAMPLES, 'Mismatch in dataset split!');

% Generate indices
All_IDX = 1:TOTAL_SAMPLES;

% Randomly select training samples
training_samples = randperm(TOTAL_SAMPLES, NUM_TRAIN).';

% Remaining indices are used for testing
testing_samples = setdiff(All_IDX, training_samples).';

% Validate the sizes
assert(length(training_samples) == NUM_TRAIN, 'Training set size mismatch!');
assert(length(testing_samples) == NUM_TEST, 'Testing set size mismatch!');

% Save the indices
save(['./samples_indices_', num2str(TOTAL_SAMPLES), '.mat'], 'training_samples', 'testing_samples');

disp('Dataset split completed successfully!');
