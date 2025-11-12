clc; clearvars; close all; warning('off','all');

%% ====== Experiment Configuration ======
METHOD_TAG = 'LSTM-MixedChannel';  % Tag for saving outputs
training_type = 'HighSNR';         % 'HighSNR' or 'MixedSNR'

%% ====== Test Channel Configuration ======
% Specify which channel you tested on
mobility = 'High';
ChType   = 'VTV_EX';      % Channel model: 'VTV_SDWW', 'VTV_EX', 'RTV_SUS', 'VTV_UC'
modu     = '16QAM';
scheme   = 'DPA_TA';      % Scheme used for classical baseline
testing_samples = 2000;   % Number of test samples per SNR

%% ====== System Parameters ======
if     isequal(modu,'QPSK'),  nBitPerSym = 2;
elseif isequal(modu,'16QAM'), nBitPerSym = 4;
elseif isequal(modu,'64QAM'), nBitPerSym = 6;
end

M   = 2 ^ nBitPerSym;                        % QAM order
Pow = mean(abs(qammod(0:(M-1),M)).^2);       % normalization

% Load simulation parameters
param_file = sprintf('./%s_%s_%s_testing_simulation_0.mat', mobility, ChType, modu);
if exist(param_file, 'file')
    load(param_file, 'Random_permutation_Vector', 'Interleaver_Rows', ...
         'Interleaver_Columns', 'scramInit', 'trellis', 'tbl', 'nBitPerSym', ...
         'nSym', 'nDSC', 'M', 'Pow');
    fprintf('Loaded simulation parameters from: %s\n', param_file);
else
    % Set default parameters if file not found
    fprintf('Parameter file not found, using defaults\n');
    nSym    = 50;
    nDSC    = 48;
    constlen = 7;
    trellis  = poly2trellis(constlen,[171 133]);
    tbl      = 34;
    scramInit = 93;
    Interleaver_Rows    = 16;
    Interleaver_Columns = (nBitPerSym * nDSC * nSym) / Interleaver_Rows;
    Random_permutation_Vector = randperm(nBitPerSym*nDSC*nSym);
end

nUSC     = 52; 
dpositions = [1:6, 8:20, 22:31, 33:45, 47:52].'; 

EbN0dB  = (0:5:40)'; 
N_SNR   = length(EbN0dB);

fprintf('======================================================\n');
fprintf('BER/NMSE Calculation for Mixed Channel LSTM\n');
fprintf('======================================================\n');
fprintf('Method: %s\n', METHOD_TAG);
fprintf('Training Type: %s\n', training_type);
fprintf('Test Channel: %s - %s\n', mobility, ChType);
fprintf('Modulation: %s\n', modu);
fprintf('Scheme: %s\n', scheme);
fprintf('Test Samples: %d\n', testing_samples);
fprintf('SNR Range: %d to %d dB\n', EbN0dB(1), EbN0dB(end));
fprintf('======================================================\n\n');

%% ====== Initialize Metrics ======
Phf     = zeros(N_SNR,1);
Err_scheme_LSTM = zeros(N_SNR,1);
Ber_scheme_LSTM = zeros(N_SNR,1);

%% ====== Process Each SNR ======
for n_snr = 1:N_SNR 
    fprintf('Processing SNR: %d dB\n', EbN0dB(n_snr));
    tic;
    
    % ===== Load simulation snapshot (ground truth) =====
    sim_file = sprintf('./%s_%s_%s_testing_simulation_%d.mat', ...
                       mobility, ChType, modu, EbN0dB(n_snr));
    
    if ~exist(sim_file, 'file')
        warning('Simulation file not found: %s (skipping)', sim_file);
        continue;
    end
    
    load(sim_file, 'True_Channels_Structure', 'Received_Symbols_FFT_Structure', ...
         'TX_Bits_Stream_Structure');
    
    % ===== Load LSTM results =====
    result_file = sprintf('./%s_%s_%s_%s_LSTM_Results_%d.mat', ...
                         mobility, ChType, modu, scheme, EbN0dB(n_snr));
    
    if ~exist(result_file, 'file')
        warning('LSTM result file not found: %s (skipping)', result_file);
        continue;
    end
    
    load(result_file);
    
    % ===== Extract data with correct variable names =====
    % LSTM results use format: {scheme}_test_y_{SNR}
    TestY_varname = sprintf('%s_test_y_%d', scheme, EbN0dB(n_snr));
    CorrectedY_varname = sprintf('%s_corrected_y_%d', scheme, EbN0dB(n_snr));
    
    if ~exist(TestY_varname, 'var')
        warning('Variable %s not found in result file', TestY_varname);
        continue;
    end
    
    TestY = eval(TestY_varname);
    scheme_LSTM = eval(CorrectedY_varname);
    
    % TestY: [samples, time, 2*nDSC] -> [nDSC, time, samples] (complex)
    TestY = permute(TestY, [3, 2, 1]);
    TestY = TestY(1:nDSC, :, :) + 1j * TestY(nDSC+1:2*nDSC, :, :);
    
    % scheme_LSTM: [samples, time, nDSC] -> [nDSC, time, samples] (complex)
    scheme_LSTM = permute(scheme_LSTM, [3, 2, 1]);
    
    % ===== Process each test sample =====
    for u = 1:size(scheme_LSTM, 3)
        H_scheme_LSTM = scheme_LSTM(:, :, u);
        
        % Calculate NMSE
        Phf(n_snr) = Phf(n_snr) + mean(sum(abs(True_Channels_Structure(dpositions, :, u)).^2)); 
        Err_scheme_LSTM(n_snr) = Err_scheme_LSTM(n_snr) + ...
            mean(sum(abs(H_scheme_LSTM - True_Channels_Structure(dpositions, :, u)).^2)); 
        
        % ===== Demodulation and BER calculation =====
        % Equalize received symbols
        equalized_symbols = Received_Symbols_FFT_Structure(dpositions, :, u) ./ H_scheme_LSTM;
        
        % QAM demodulation
        Bits_scheme_LSTM = de2bi(qamdemod(sqrt(Pow) * equalized_symbols, M));
        
        % Deinterleaving and decoding
        deinterleaved_bits = deintrlv(Bits_scheme_LSTM(:), Random_permutation_Vector);
        deinterleaved_matrix = matintrlv(deinterleaved_bits.', Interleaver_Columns, Interleaver_Rows).';
        decoded_bits = vitdec(deinterleaved_matrix, trellis, tbl, 'trunc', 'hard');
        descrambled_bits = wlanScramble(decoded_bits, scramInit);
        
        % Calculate bit errors
        Ber_scheme_LSTM(n_snr) = Ber_scheme_LSTM(n_snr) + ...
            biterr(descrambled_bits, TX_Bits_Stream_Structure(:, u));
    end
    
    elapsed = toc;
    fprintf('  Completed in %.2f seconds\n', elapsed);
end

%% ====== Normalize Metrics ======
Phf = Phf ./ testing_samples;
ERR_scheme_LSTM = Err_scheme_LSTM ./ (testing_samples * Phf); 
BER_scheme_LSTM = Ber_scheme_LSTM  ./ (testing_samples * nSym * nDSC * nBitPerSym);

% Convert NMSE to dB
NMSE_dB = 10 * log10(ERR_scheme_LSTM);

%% ====== Display Results ======
fprintf('\n======================================================\n');
fprintf('Results Summary\n');
fprintf('======================================================\n');
fprintf('SNR (dB) | NMSE (dB) | BER\n');
fprintf('---------|-----------|----------\n');
for n_snr = 1:N_SNR
    fprintf('%8d | %9.2f | %.4e\n', EbN0dB(n_snr), NMSE_dB(n_snr), BER_scheme_LSTM(n_snr));
end
fprintf('======================================================\n\n');

%% ====== Save Metrics ======
output_filename = sprintf('metrics_%s_%s_%s_%s.mat', METHOD_TAG, training_type, mobility, ChType);
save(output_filename, ...
     'EbN0dB', 'Phf', 'ERR_scheme_LSTM', 'NMSE_dB', 'BER_scheme_LSTM', ...
     'mobility', 'ChType', 'modu', 'scheme', 'nSym', 'nDSC', 'METHOD_TAG', ...
     'training_type', 'testing_samples');

fprintf('Metrics saved to: %s\n\n', output_filename);

%% ====== Plot NMSE ======
figure('Position', [100, 100, 800, 600]);
plot(EbN0dB, NMSE_dB, 'r-s', 'LineWidth', 2, 'MarkerSize', 8); 
grid on;
xlabel('SNR (dB)', 'FontSize', 12, 'FontWeight', 'bold'); 
ylabel('NMSE (dB)', 'FontSize', 12, 'FontWeight', 'bold');
title(sprintf('NMSE vs SNR — %s (%s on %s-%s)', METHOD_TAG, training_type, mobility, ChType), ...
      'FontSize', 14, 'FontWeight', 'bold');
set(gca, 'FontSize', 11);

% Save NMSE plot
nmse_filename = sprintf('NMSE_%s_%s_%s_%s.png', METHOD_TAG, training_type, mobility, ChType);
saveas(gcf, nmse_filename);
fprintf('NMSE plot saved to: %s\n', nmse_filename);

%% ====== Plot BER ======
figure('Position', [120, 120, 800, 600]);
semilogy(EbN0dB, BER_scheme_LSTM, 'b-d', 'LineWidth', 2, 'MarkerSize', 8); 
grid on;
xlabel('SNR (dB)', 'FontSize', 12, 'FontWeight', 'bold'); 
ylabel('BER', 'FontSize', 12, 'FontWeight', 'bold');
title(sprintf('BER vs SNR — %s (%s on %s-%s)', METHOD_TAG, training_type, mobility, ChType), ...
      'FontSize', 14, 'FontWeight', 'bold');
set(gca, 'FontSize', 11);
ylim([1e-6 1]);

% Save BER plot
ber_filename = sprintf('BER_%s_%s_%s_%s.png', METHOD_TAG, training_type, mobility, ChType);
saveas(gcf, ber_filename);
fprintf('BER plot saved to: %s\n', ber_filename);

fprintf('\n======================================================\n');
fprintf('Analysis Complete!\n');
fprintf('======================================================\n');