function gen_mixed_channel_testing()
clc; clearvars; close all; warning('off','all');

%% ----------------- Helper (channel toolbox) -----------------
ch_func = Channel_functions();

%% ----------------- OFDM / 802.11p parameters ----------------
ofdmBW   = 10e6;
nFFT     = 64;
nDSC     = 48;
nPSC     = 4;
nZSC     = 12;
nUSC     = nDSC + nPSC;
K        = nUSC + nZSC;
nSym     = 50;
deltaF   = ofdmBW/nFFT;
Tfft     = 1/deltaF;
Tgi      = Tfft/4;
K_cp     = nFFT*Tgi/Tfft;

% Subcarrier locations (1..64) in the IFFT grid
pilots_locations = [8,22,44,58].';
pilots           = [1 1 1 -1].';
data_locations   = [2:7, 9:21, 23:27, 39:43, 45:57, 59:64].';
null_locations   = [1, 28:38].';

% Positions within Kset
ppositions = [7,21,32,46].';
dpositions = [1:6, 8:20, 22:31, 33:45, 47:52].';

%% ----------------- IEEE 802.11p preamble ------------
dp = [ 0 0 0 0 0 0 +1 +1 -1 -1 +1 +1 -1 +1 -1 +1 +1 +1 +1 +1 +1 -1 -1 +1 +1 -1 +1 -1 +1 +1 +1 +1 ...
       0 +1 -1 -1 +1 +1 -1 +1 -1 +1 -1 -1 -1 -1 -1 +1 +1 -1 -1 +1 -1 +1 -1 +1 +1 +1 +1 0 0 0 0 0];

Ep                  = 1;
dp                  = fftshift(dp);
predefined_preamble = dp;
Kset                = find(dp~=0);
Kon                 = length(Kset);
dp                  = sqrt(Ep)*dp.';
xp                  = sqrt(K)*ifft(dp);
xp_cp               = [xp(end-K_cp+1:end); xp];
preamble_80211p     = repmat(xp_cp,1,2);

%% ----------------- Modulation -----------------
modu     = '16QAM';
Mod_Type = 1;

if(strcmp(modu,'QPSK') == 1)
    nBitPerSym = 2;
elseif (strcmp(modu,'16QAM') == 1)
    nBitPerSym = 4;
elseif (strcmp(modu,'64QAM') == 1)
    nBitPerSym = 6;
end
M   = 2^nBitPerSym;
Pow = mean(abs(qammod(0:M-1, M)).^2);

%% ----------------- PHY (scrambler/coder/interleaver) ---------
scramInit = 93;
constlen  = 7;
trellis   = poly2trellis(constlen,[171 133]);
tbl       = 34;
rate      = 1/2;

Interleaver_Rows    = 16;
Interleaver_Columns = (nBitPerSym * nDSC * nSym) / Interleaver_Rows;
Random_permutation_Vector = randperm(nBitPerSym*nDSC*nSym);

%% ----------------- Channel models configuration -----------------
fc = 5.9e9;
c = 3e8;
fs = K*deltaF;

% Define ALL 6 channel models for testing
% 3 SEEN (trained on) + 3 UNSEEN (not trained on)
channel_configs = {
    % SEEN channels (used in training)
    struct('model', 'VTV_SDWW', 'vel_kmh', 104, 'type', 'seen');
    struct('model', 'VTV_EX',   'vel_kmh', 104, 'type', 'seen');
    struct('model', 'RTV_SUS',  'vel_kmh', 40,  'type', 'seen');
    
    % UNSEEN channels (for OOD evaluation)
    struct('model', 'VTV_UC',   'vel_kmh', 40, 'type', 'unseen');  % Urban Canyon
    struct('model', 'RTV_EX',   'vel_kmh', 104,  'type', 'unseen');  % Highway RSU
    struct('model', 'RTV_UC',  'vel_kmh', 40,  'type', 'unseen');  % Suburban
};

N_MODELS = length(channel_configs);

% STA parameters
alpha  = 2;
Beta   = 2;
w      = 1 / (2*Beta + 1);
lambda = -Beta:Beta;

%% ----------------- SNR configuration ------------------
% Testing: always test on ALL SNRs
EbN0dB = 0:5:40;

SNR_p  = EbN0dB + 10*log10(K/nDSC) + 10*log10(K/(K+K_cp)) + 10*log10(nBitPerSym) + 10*log10(rate);
SNR_p  = SNR_p(:);
N0     = Ep*10.^(-SNR_p/10);
N_SNR  = length(SNR_p);

%% ----------------- Load sample indices ------------------
load('./samples_indices_4000.mat', 'testing_samples');
indices = testing_samples;
N_SAMPLES = size(indices, 1);  % 2000 test samples per channel

fprintf('\n========================================\n');
fprintf('GENERATING TESTING DATA\n');
fprintf('========================================\n');
fprintf('Channel models: %d (3 seen + 3 unseen)\n', N_MODELS);
fprintf('Samples per channel: %d\n', N_SAMPLES);
fprintf('SNR levels: %d\n', N_SNR);
fprintf('Total files: %d (channels) × %d (SNRs) = %d\n', N_MODELS, N_SNR, N_MODELS * N_SNR);
fprintf('========================================\n\n');

%% ----------------- Main loop: Channel → SNR → Realizations --------
for n_model = 1:N_MODELS
    config = channel_configs{n_model};
    model_name = config.model;
    vel_kmh = config.vel_kmh;
    channel_type = config.type;
    
    % Calculate Doppler
    fD = (vel_kmh/3.6)/c * fc;
    
    % Determine mobility label
    if vel_kmh >= 80
        mobility = 'Very_High';
    else
        mobility = 'High';
    end
    
    fprintf('\n========================================\n');
    fprintf('Channel %d/%d: %s (%s)\n', n_model, N_MODELS, model_name, channel_type);
    fprintf('========================================\n');
    fprintf('Velocity: %d km/h\n', vel_kmh);
    fprintf('Doppler: %.1f Hz\n', fD);
    fprintf('Mobility: %s\n', mobility);
    fprintf('========================================\n\n');
    
    % Generate channel object
    rchan = ch_func.GenFadingChannel(model_name, fD, fs);
    
    % Loop over SNR levels
    for n_snr = 1:N_SNR
        fprintf('  SNR = %2d dB ... ', EbN0dB(n_snr));
        tic;
        
        % Allocate structures for this channel/SNR
        nInfoBits = nDSC * nSym * nBitPerSym * rate;
        
        TX_Bits_Stream_Structure       = single(zeros(nInfoBits, N_SAMPLES));
        Received_Symbols_FFT_Structure = single(zeros(Kon, nSym, N_SAMPLES));
        True_Channels_Structure        = single(zeros(Kon, nSym, N_SAMPLES));
        HLS_Structure                  = single(zeros(Kon, N_SAMPLES));
        DPA_Structure                  = single(zeros(Kon, nSym, N_SAMPLES));
        DPA_TA_Structure               = single(zeros(Kon, nSym, N_SAMPLES));
        STA_Structure                  = single(zeros(Kon, nSym, N_SAMPLES));
        TRFI_Structure                 = single(zeros(Kon, nSym, N_SAMPLES));
        CDP_Structure                  = single(zeros(nDSC, nSym, N_SAMPLES));
        R_Symbols_Training_Structure   = single(zeros(Kon, nSym, N_SAMPLES));
        
        % Loop over realizations
        for n_ch = 1:N_SAMPLES
            % -------- Bit source -----
            Bits_Stream_Coded = randi(2, nInfoBits, 1) - 1;

            % -------- Tx chain ---------------
            scrambledData = wlanScramble(Bits_Stream_Coded, scramInit);
            dataEnc       = convenc(scrambledData, trellis);
            
            codedata = dataEnc.';
            Matrix_Interleaved_Data   = matintrlv(codedata, Interleaver_Rows, Interleaver_Columns).';
            General_Block_Interleaved = intrlv(Matrix_Interleaved_Data, Random_permutation_Vector);

            % -------- M-QAM mapping ----------------------
            TxBits_Coded = reshape(General_Block_Interleaved, nDSC, nSym, nBitPerSym);
            TxData_Coded = zeros(nDSC, nSym);
            for m = 1:nBitPerSym
                TxData_Coded = TxData_Coded + TxBits_Coded(:,:,m) * 2^(m-1);
            end
            Modulated_Bits = (1/sqrt(Pow)) * qammod(TxData_Coded, M);

            % -------- Build FD grid and go to TD -------------------
            OFDM_Frame_Coded                     = zeros(K, nSym);
            OFDM_Frame_Coded(data_locations,:)   = Modulated_Bits;
            OFDM_Frame_Coded(pilots_locations,:) = repmat(pilots, 1, nSym);

            IFFT_Data_Coded = sqrt(K) * ifft(OFDM_Frame_Coded);
            CP_Coded = IFFT_Data_Coded((K - K_cp + 1):K, :);
            IFFT_Data_CP_Coded = [CP_Coded; IFFT_Data_Coded];
            IFFT_Data_CP_Preamble_Coded = [preamble_80211p, IFFT_Data_CP_Coded];

            % -------- Channel --------
            release(rchan);
            rchan.Seed = indices(n_ch, 1);
            [h, y] = ch_func.ApplyChannel(rchan, IFFT_Data_CP_Preamble_Coded, K_cp);
            
            yp = y((K_cp+1):end, 1:2);
            y  = y((K_cp+1):end, 3:end);
            
            yFD = sqrt(1/K) * fft(y);
            yfp = sqrt(1/K) * fft(yp);
            
            h  = h((K_cp+1):end, :);
            hf = fft(h);
            hfp1 = hf(:, 1);
            hfp2 = hf(:, 2);
            hfp  = (hfp1 + hfp2) / 2;
            hf   = hf(:, 3:end);

            % -------- Add noise --------
            noise_preamble     = sqrt(N0(n_snr)) * ch_func.GenRandomNoise([K, 2], 1);
            noise_OFDM_Symbols = sqrt(N0(n_snr)) * ch_func.GenRandomNoise([K, size(yFD,2)], 1);
            
            yfp_r = yfp + noise_preamble;
            y_r   = yFD + noise_OFDM_Symbols;

            % -------- LS estimation -----------------------
            he_LS_Preamble = ((yfp_r(Kset,1) + yfp_r(Kset,2)) ./ (2 .* predefined_preamble(Kset).'));

            % -------- Channel estimators --------
            [H_DPA, ~] = DPA(he_LS_Preamble, y_r, Kset, ppositions, modu, nUSC, nSym);
            [H_DPA_TA, ~] = DPA_TA(he_LS_Preamble, y_r, Kset, modu, nUSC, nSym, ppositions);
            [H_STA, ~] = STA(he_LS_Preamble, y_r, Kset, modu, nUSC, nSym, ppositions, alpha, w, lambda);
            [H_TRFI, ~] = TRFI(he_LS_Preamble, y_r, Kset, yfp_r(Kset,2), ...
                               predefined_preamble(1,Kset).', ppositions, modu, nUSC, nSym);
            
            % CDP (data-only)
            H_CDP = H_DPA(dpositions, :);

            % -------- Store structures --------
            TX_Bits_Stream_Structure(:, n_ch)         = single(Bits_Stream_Coded);
            Received_Symbols_FFT_Structure(:, :, n_ch) = single(y_r(Kset,:));
            True_Channels_Structure(:, :, n_ch)       = single(hf(Kset,:));
            HLS_Structure(:, n_ch)                    = single(he_LS_Preamble(:));
            DPA_Structure(:, :, n_ch)                 = single(H_DPA);
            DPA_TA_Structure(:, :, n_ch)              = single(H_DPA_TA);
            STA_Structure(:, :, n_ch)                 = single(H_STA);
            TRFI_Structure(:, :, n_ch)                = single(H_TRFI);
            CDP_Structure(:, :, n_ch)                 = single(H_CDP);
            R_Symbols_Training_Structure(:, :, n_ch)  = single(y_r(Kset,:));
        end

        % -------- Save file ------------------------
        filename = sprintf('./%s_%s_%s_testing_simulation_%d.mat', ...
                           mobility, model_name, modu, EbN0dB(n_snr));
                           
        save(filename, ...
             'TX_Bits_Stream_Structure', ...
             'Received_Symbols_FFT_Structure', ...
             'True_Channels_Structure', ...
             'HLS_Structure', ...
             'DPA_Structure', ...
             'DPA_TA_Structure', ...
             'STA_Structure', ...
             'TRFI_Structure', ...
             'CDP_Structure', ...
             'R_Symbols_Training_Structure', ...
             'Random_permutation_Vector', ...
             'Kset', 'ppositions', 'dpositions', ...
             'nBitPerSym', 'nDSC', 'nSym', 'Pow', 'M', ...
             'scramInit', 'tbl', 'trellis', ...
             'Interleaver_Rows', 'Interleaver_Columns', ...
             '-v7.3');
             
        fprintf('Done (%.1fs, %.1f MB)\n', toc, dir(filename).bytes/1e6);
    end
end

fprintf('\n========================================\n');
fprintf('TESTING DATA GENERATION COMPLETE!\n');
fprintf('========================================\n');
fprintf('Generated %d files total\n', N_MODELS * N_SNR);
fprintf('\nFiles organized by channel:\n');
for n_model = 1:N_MODELS
    config = channel_configs{n_model};
    if config.vel_kmh >= 80
        mobility = 'Very_High';
    else
        mobility = 'High';
    end
    fprintf('  %s_%s_%s_testing_simulation_*.mat (%s)\n', ...
            mobility, config.model, modu, config.type);
end
fprintf('========================================\n\n');

end % function
