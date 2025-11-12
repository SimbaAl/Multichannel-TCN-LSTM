clc; clearvars; close all; warning('off','all');

% Load pre-defined DNN Testing Indices
load('./samples_indices_18000.mat');

configuration = 'training'; % 'training' or 'testing'

% Define Simulation parameters
nSC_In = 104;
nSC_Out = 96;
nSym = 50;
modu = '16QAM';
scheme = 'DPA_TA';

% Define channel models (must match generation script)
channel_models = {'VTV_SDWW', 'VTV_EX', 'RTV_SUS'};
n_channels = length(channel_models);

% Pilot and data positions
ppositions = [7, 21, 32, 46].';
dpositions = [1:6, 8:20, 22:31, 33:45, 47:52].';

% SNR configuration
if isequal(configuration, 'training')
    indices = training_samples;
    
    % CORRECTED: Match the generation script's SNR strategy
    % Option 1: HighSNR (LSTM-DPA-TA, Dual-cell LSTM)
    EbN0dB = 40;
    snr_label = 'HighSNR';
    
    % Option 2: MixedSNR (TCN-DPA, CNN-Transformer)
    % EbN0dB = 0:5:40;
    % snr_label = 'MixedSNR';
    
elseif isequal(configuration, 'testing')
    indices = testing_samples;
    EbN0dB = 0:5:40;
end

Dataset_size = size(indices, 1);
SNR = EbN0dB.';
N_SNR = length(SNR);

fprintf('======================================================\n');
fprintf('LSTM Mixed Channel Data Preprocessing\n');
fprintf('======================================================\n');
fprintf('Configuration: %s\n', configuration);
fprintf('Modulation: %s\n', modu);
fprintf('Scheme: %s\n', scheme);

%% ===================================================================
%% TRAINING: Load from MixedChannel files
%% ===================================================================
if isequal(configuration, 'training')
    fprintf('SNR strategy: %s\n', snr_label);
    fprintf('SNR values: [%s] dB\n', num2str(EbN0dB));
    fprintf('======================================================\n\n');
    
    fprintf('Loading MIXED CHANNEL training data...\n\n');
    
    % CORRECTED: Load from the MixedChannel_*.mat files generated
    % Calculate total samples across all SNRs
    total_samples = 0;
    for n_snr = 1:N_SNR
        % Load to get size
        filename = sprintf('./MixedChannel_%s_%s_training_simulation_%d.mat', ...
                          snr_label, modu, EbN0dB(n_snr));
        
        if ~exist(filename, 'file')
            error('Training file not found: %s\nRun gen_mixed_channel_training.m first!', filename);
        end
        
        temp = matfile(filename);
        sz = size(temp, 'True_Channels_Structure');
        total_samples = total_samples + sz(3);
    end
    
    fprintf('Total samples across all SNRs: %d\n', total_samples);
    
    % Pre-allocate combined arrays
    All_Dataset_X = zeros(total_samples, nSym, nSC_In);
    All_Dataset_Y = zeros(total_samples, nSym, nSC_Out);
    All_Channel_Labels = cell(total_samples, 1);
    All_SNR_Labels = zeros(total_samples, 1);
    
    sample_idx = 1;
    
    % Load and process each SNR file
    for n_snr = 1:N_SNR
        fprintf('Processing SNR: %d dB\n', EbN0dB(n_snr));
        
        filename = sprintf('./MixedChannel_%s_%s_training_simulation_%d.mat', ...
                          snr_label, modu, EbN0dB(n_snr));
        
        load(filename, 'True_Channels_Structure', [scheme '_Structure'], ...
             'HLS_Structure', 'Channel_Model_Labels');
        
        % Get number of samples in this file
        n_samples = size(True_Channels_Structure, 3);
        
        % Remove first symbol (guard/preamble)
        %True_Channels_Structure = True_Channels_Structure(:, 2:end, :);
        
        % Get scheme channels
        scheme_Channels_Structure = eval([scheme '_Structure']);
        
        RHP = real(scheme_Channels_Structure(:, 1:end-1, :));
        IHP = imag(scheme_Channels_Structure(:, 1:end-1, :));
        
        % Prepare X (Input): [real(HLS); imag(HLS)] at t=1, then [real(H); imag(H)] for t=2:50
        Dataset_X = zeros(nSC_In, nSym, n_samples);
        Dataset_X(:, 1, :) = [real(HLS_Structure); imag(HLS_Structure)];
        Dataset_X(:, 2:end, :) = [RHP; IHP];
        
        % Prepare Y (Output): Ground truth channels at data subcarriers only
        Dataset_Y = zeros(nSC_Out, nSym, n_samples);
        Dataset_Y(1:48, :, :) = real(True_Channels_Structure(dpositions, :, :));
        Dataset_Y(49:96, :, :) = imag(True_Channels_Structure(dpositions, :, :));
        
        % Permute to (samples, time, features)
        Dataset_X = permute(Dataset_X, [3, 2, 1]);
        Dataset_Y = permute(Dataset_Y, [3, 2, 1]);
        
        % Store in combined arrays
        end_idx = sample_idx + n_samples - 1;
        All_Dataset_X(sample_idx:end_idx, :, :) = Dataset_X;
        All_Dataset_Y(sample_idx:end_idx, :, :) = Dataset_Y;
        
        % Store labels
        for s = 1:n_samples
            All_Channel_Labels{sample_idx + s - 1} = char(Channel_Model_Labels(s));
            All_SNR_Labels(sample_idx + s - 1) = EbN0dB(n_snr);
        end
        
        sample_idx = end_idx + 1;
        
        fprintf('  Loaded %d samples\n', n_samples);
    end
    
    % Shuffle the combined dataset
    fprintf('\nShuffling combined dataset...\n');
    shuffle_idx = randperm(total_samples);
    All_Dataset_X = All_Dataset_X(shuffle_idx, :, :);
    All_Dataset_Y = All_Dataset_Y(shuffle_idx, :, :);
    All_Channel_Labels = All_Channel_Labels(shuffle_idx);
    All_SNR_Labels = All_SNR_Labels(shuffle_idx);
    
    % Save mixed training data
    LSTM_Datasets.Train_X = All_Dataset_X;
    LSTM_Datasets.Train_Y = All_Dataset_Y;
    LSTM_Datasets.Channel_Labels = All_Channel_Labels;
    LSTM_Datasets.SNR_Labels = All_SNR_Labels;
    
    output_filename = sprintf('./MixedChannel_%s_%s_%s_LSTM_training_dataset.mat', ...
                             snr_label, modu, scheme);
    save(output_filename, 'LSTM_Datasets', '-v7.3');
    
    fprintf('\n======================================================\n');
    fprintf('Training data saved: %s\n', output_filename);
    fprintf('Total samples: %d\n', total_samples);
    fprintf('Shape X: (%d, %d, %d)\n', size(All_Dataset_X));
    fprintf('Shape Y: (%d, %d, %d)\n', size(All_Dataset_Y));
    
    % Display channel distribution
    fprintf('\nChannel distribution:\n');
    for ch_idx = 1:n_channels
        count = sum(strcmp(All_Channel_Labels, channel_models{ch_idx}));
        fprintf('  %s: %d samples (%.1f%%)\n', ...
                channel_models{ch_idx}, count, 100*count/total_samples);
    end
    
    % Display SNR distribution
    fprintf('\nSNR distribution:\n');
    unique_snrs = unique(All_SNR_Labels);
    for i = 1:length(unique_snrs)
        count = sum(All_SNR_Labels == unique_snrs(i));
        fprintf('  %d dB: %d samples (%.1f%%)\n', ...
                unique_snrs(i), count, 100*count/total_samples);
    end
    
    fprintf('======================================================\n');

%% ===================================================================
%% TESTING: Load from individual channel files
%% ===================================================================
elseif isequal(configuration, 'testing')
    fprintf('======================================================\n\n');
    fprintf('Generating testing data for each channel separately...\n\n');
    
    % CORRECTED: For testing, we still need individual channel files
    % These should be generated by a separate testing script
    
    % Loop through each channel model
    for ch_idx = 1:n_channels
        ChType = channel_models{ch_idx};
        fprintf('Processing channel: %s\n', ChType);
        
        % Determine mobility label based on channel type
        if contains(ChType, 'RTV')
            mobility = 'High';
        else
            mobility = 'Very_High';
        end
        
        % Loop through each SNR
        for n_snr = 1:N_SNR
            fprintf('  SNR: %d dB\n', EbN0dB(n_snr));
            
            % CORRECTED: Load from individual testing files
            % Format: Very_High_VTV_SDWW_16QAM_testing_simulation_20.mat
            filename = sprintf('./%s_%s_%s_testing_simulation_%d.mat', ...
                              mobility, ChType, modu, EbN0dB(n_snr));
            
            if ~exist(filename, 'file')
                warning('File not found: %s (skipping)', filename);
                continue;
            end
            
            load(filename, 'True_Channels_Structure', [scheme '_Structure'], ...
                'HLS_Structure', 'Received_Symbols_FFT_Structure');
            
            % Remove first symbol (guard)
            True_Channels_Structure = True_Channels_Structure(:, 2:end, :);
            
            % Get number of samples
            n_samples = size(True_Channels_Structure, 3);
            
            % Get scheme channels
            scheme_Channels_Structure = eval([scheme '_Structure']);
            
            RPP = real(scheme_Channels_Structure(ppositions, 1:end-1, :));
            IPP = imag(scheme_Channels_Structure(ppositions, 1:end-1, :));
            
            % Extract received symbols at data subcarriers
            Received_Symbols_FFT_Structure = Received_Symbols_FFT_Structure(dpositions, :, :);
            
            % Prepare X (Input): sparse format with pilots only
            Dataset_X = zeros(nSC_In, nSym, n_samples);
            Dataset_X(:, 1, :) = [real(HLS_Structure); imag(HLS_Structure)];
            Dataset_X(ppositions, 2:end, :) = RPP;
            Dataset_X(ppositions + 52, 2:end, :) = IPP;
            
            % Prepare Y (Output): Ground truth at data subcarriers
            Dataset_Y = zeros(nSC_Out, nSym, n_samples);
            Dataset_Y(1:48, :, :) = real(True_Channels_Structure(dpositions, :, :));
            Dataset_Y(49:96, :, :) = imag(True_Channels_Structure(dpositions, :, :));
            
            % Permute to (samples, time, features)
            Dataset_X = permute(Dataset_X, [3, 2, 1]);
            Dataset_Y = permute(Dataset_Y, [3, 2, 1]);
            Received_Symbols_FFT_Structure = permute(Received_Symbols_FFT_Structure, [3, 2, 1]);
            
            % Save per-channel, per-SNR test data
            LSTM_Datasets = struct();
            LSTM_Datasets.Test_X = Dataset_X;
            LSTM_Datasets.Test_Y = Dataset_Y;
            LSTM_Datasets.Y_DataSubCarriers = Received_Symbols_FFT_Structure;
            LSTM_Datasets.Channel_Label = ChType;
            LSTM_Datasets.Mobility = mobility;
            LSTM_Datasets.SNR_dB = EbN0dB(n_snr);
            
            output_filename = sprintf('./%s_%s_%s_%s_LSTM_testing_dataset_%d.mat', ...
                                     mobility, ChType, modu, scheme, EbN0dB(n_snr));
            save(output_filename, 'LSTM_Datasets');
            
            fprintf('    Saved: %s\n', output_filename);
        end
        fprintf('\n');
    end
    
    fprintf('======================================================\n');
    fprintf('Testing data generation complete!\n');
    fprintf('Saved %d test files (%d channels × %d SNRs)\n', ...
            n_channels * N_SNR, n_channels, N_SNR);
    fprintf('======================================================\n');
end

fprintf('\nDONE!\n');
