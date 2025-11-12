clc; clearvars; close all; warning('off','all');

% Define Simulation parameters
nSC_In = 104;
nSC_Out = 96;
nSym = 50;
modu = '16QAM';
scheme = 'DPA_TA';

% Define channel models with their mobility (must match your generation script)
% Format: {mobility, model_name}
channel_configs = {
    'Very_High', 'VTV_SDWW';
    'Very_High', 'VTV_EX';
    'High', 'RTV_SUS';
    'High', 'VTV_UC'  % Add if you have this
};
n_channels = size(channel_configs, 1);

% Pilot and data positions
ppositions = [7, 21, 32, 46].';
dpositions = [1:6, 8:20, 22:31, 33:45, 47:52].';

% Testing SNR values
EbN0dB = 0:5:40;
N_SNR = length(EbN0dB);

fprintf('======================================================\n');
fprintf('LSTM Mixed Channel - Testing Data Preprocessing\n');
fprintf('======================================================\n');
fprintf('Modulation: %s\n', modu);
fprintf('Scheme: %s\n', scheme);
fprintf('SNR values: [%s] dB\n', num2str(EbN0dB));
fprintf('======================================================\n\n');

% Loop through each channel model
for ch_idx = 1:n_channels
    mobility = channel_configs{ch_idx, 1};
    model = channel_configs{ch_idx, 2};
    
    fprintf('Processing channel: %s - %s\n', mobility, model);
    
    % Loop through each SNR
    for n_snr = 1:N_SNR
        fprintf('  SNR: %d dB\n', EbN0dB(n_snr));
        
        % Load raw testing simulation file
        % Format: High_VTV_UC_16QAM_testing_simulation_20.mat
        filename = sprintf('./%s_%s_%s_testing_simulation_%d.mat', ...
                          mobility, model, modu, EbN0dB(n_snr));
        
        if ~exist(filename, 'file')
            warning('File not found: %s (skipping)', filename);
            continue;
        end
        
        fprintf('  Loading: %s\n', filename);
        load(filename, 'True_Channels_Structure', [scheme '_Structure'], ...
            'HLS_Structure', 'Received_Symbols_FFT_Structure');
        
        % Remove first symbol (guard/preamble) if needed
        % Uncomment the line below if your data has a guard symbol
        % True_Channels_Structure = True_Channels_Structure(:, 2:end, :);
        
        % Get number of samples
        n_samples = size(True_Channels_Structure, 3);
        
        % Get scheme channels
        scheme_Channels_Structure = eval([scheme '_Structure']);
        
        % Extract real and imaginary parts at pilot positions
        RPP = real(scheme_Channels_Structure(ppositions, 1:end-1, :));
        IPP = imag(scheme_Channels_Structure(ppositions, 1:end-1, :));
        
        % Extract received symbols at data subcarriers
        Received_Symbols_FFT_Structure = Received_Symbols_FFT_Structure(dpositions, :, :);
        
        % Prepare X (Input): sparse format with pilots only
        % First symbol: [real(HLS); imag(HLS)]
        % Remaining symbols: pilots at their positions, zeros elsewhere
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
        LSTM_Datasets.Channel_Label = model;
        LSTM_Datasets.Mobility = mobility;
        LSTM_Datasets.SNR_dB = EbN0dB(n_snr);
        
        % Save with proper naming convention
        % Format: Very_High_VTV_SDWW_16QAM_DPA_TA_LSTM_testing_dataset_20.mat
        output_filename = sprintf('./%s_%s_%s_%s_LSTM_testing_dataset_%d.mat', ...
                                 mobility, model, modu, scheme, EbN0dB(n_snr));
        save(output_filename, 'LSTM_Datasets', '-v7.3');
        
        fprintf('    Saved: %s\n', output_filename);
    end
    fprintf('\n');
end

fprintf('======================================================\n');
fprintf('Testing data preprocessing complete!\n');
fprintf('Saved %d test files (%d channels × %d SNRs)\n', ...
        n_channels * N_SNR, n_channels, N_SNR);
fprintf('======================================================\n');

fprintf('\nDONE!\n');