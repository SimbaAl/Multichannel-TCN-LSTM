function gen_mixed_channel_training()
clc; clearvars; close all; warning('off','all');

%% ----------------- Helper (channel toolbox) -----------------
ch_func = Channel_functions();

%% ----------------- OFDM / 802.11p parameters ----------------
ofdmBW   = 10e6;                  % Hz
nFFT     = 64;                    % FFT size
nDSC     = 48;                    % data subcarriers
nPSC     = 4;                     % pilot subcarriers
nZSC     = 12;                    % nulls/zeros
nUSC     = nDSC + nPSC;           % used subcarriers
K        = nUSC + nZSC;           % total subcarriers (64)
nSym     = 50;                    % OFDM symbols/frame
deltaF   = ofdmBW/nFFT;           % subcarrier spacing
Tfft     = 1/deltaF;              % 6.4 us
Tgi      = Tfft/4;                % 1.6 us
K_cp     = nFFT*Tgi/Tfft;         % CP samples (16)

% Subcarrier locations (1..64) in the IFFT grid
pilots_locations = [8,22,44,58].';
pilots           = [1 1 1 -1].';
data_locations   = [2:7, 9:21, 23:27, 39:43, 45:57, 59:64].';
null_locations   = [1, 28:38].';

% Positions within Kset (used in some estimators)
ppositions = [7,21,32,46].';
dpositions = [1:6, 8:20, 22:31, 33:45, 47:52].';

%% ----------------- IEEE 802.11p preamble (robust) ------------
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
if Mod_Type==0
    nBitPerSym = 1;  
    Pow = 1; 
    M = 1;
else
    if(strcmp(modu,'QPSK') == 1)
        nBitPerSym = 2; 
    elseif (strcmp(modu,'16QAM') == 1)
        nBitPerSym = 4; 
    elseif (strcmp(modu,'64QAM') == 1)
        nBitPerSym = 6; 
    end
    M   = 2^nBitPerSym;
    Pow = mean(abs(qammod(0:M-1, M)).^2);
end

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

% Define 3 channel models with their velocities
channel_configs = {
    struct('model', 'VTV_SDWW', 'vel_kmh', 104);  % V2V Same Direction With Wall
    struct('model', 'VTV_EX',   'vel_kmh', 104);  % V2V Expressway
    struct('model', 'RTV_SUS',   'vel_kmh', 40);   % R2V Street Scene
};

N_MODELS = length(channel_configs);

% STA parameters
alpha  = 2;
Beta   = 2;
w      = 1 / (2*Beta + 1);
lambda = -Beta:Beta;

%% ----------------- SNR configuration ------------------

% Option 1: Mixed SNR (for TCN-DPA, CNN-Transformer)
EbN0dB_mixed = 0:5:40;  % All SNR levels

% Option 2: High SNR only (for LSTM-DPA-TA, Dual-cell LSTM)
EbN0dB_high = 40;       % Single high SNR

% Choose which SNR strategy to use:
USE_MIXED_SNR = true;  % Set to false for high SNR only

if USE_MIXED_SNR
    EbN0dB = EbN0dB_mixed;
    snr_label = 'MixedSNR';
    fprintf('=== Using MIXED SNR training (0-40 dB) ===\n');
else
    EbN0dB = EbN0dB_high;
    snr_label = 'HighSNR';
    fprintf('=== Using HIGH SNR training (40 dB only) ===\n');
end

SNR_p  = EbN0dB + 10*log10(K/nDSC) + 10*log10(K/(K+K_cp)) + 10*log10(nBitPerSym) + 10*log10(rate);
SNR_p  = SNR_p(:);
N0     = Ep*10.^(-SNR_p/10);
N_SNR  = length(SNR_p);

%% ----------------- Load sample indices ------------------
%load('./samples_indices_18000.mat', 'training_samples'); uncomment this for LSTM
load('./samples_indices_4000.mat', 'training_samples'); %for TCN
indices = training_samples;
N_CH_PER_MODEL = size(indices, 1);  % Samples per channel model

% Total samples = N_CH_PER_MODEL * N_MODELS * N_SNR
% For mixed SNR: 2000 * 3 * 9 = 54,000 samples
% For high SNR:  18000 * 3 * 1 = 54,000 samples

fprintf('=== Generating MIXED CHANNEL training data ===\n');
fprintf('Channel models: %d\n', N_MODELS);
fprintf('Samples per model: %d\n', N_CH_PER_MODEL);
fprintf('SNR levels: %d\n', N_SNR);
fprintf('Total samples: %d\n', N_CH_PER_MODEL * N_MODELS * N_SNR);

%% ----------------- Main loop: SNR → Channel Model → Realizations --------
for n_snr = 1:N_SNR
    fprintf('\n=== Processing SNR = %2d dB ===\n', EbN0dB(n_snr));
    tic;
    
    % Allocate COMBINED structures for ALL channel models
    total_samples = N_CH_PER_MODEL * N_MODELS;
    
    nInfoBits = nDSC * nSym * nBitPerSym * rate;
    
    TX_Bits_Stream_Structure       = single(zeros(nInfoBits, total_samples));
    Received_Symbols_FFT_Structure = single(zeros(Kon, nSym, total_samples));
    True_Channels_Structure        = single(zeros(Kon, nSym, total_samples));
    HLS_Structure                  = single(zeros(Kon, total_samples));
    DPA_Structure                  = single(zeros(Kon, nSym, total_samples));
    DPA_TA_Structure               = single(zeros(Kon, nSym, total_samples));
    STA_Structure                  = single(zeros(Kon, nSym, total_samples));
    TRFI_Structure                 = single(zeros(Kon, nSym, total_samples));
    R_Symbols_Training_Structure   = single(zeros(Kon, nSym, total_samples));
    
    % Metadata: store which channel model each sample came from
    Channel_Model_Labels = strings(total_samples, 1);
    
    % Loop over channel models
    for n_model = 1:N_MODELS
        config = channel_configs{n_model};
        model_name = config.model;
        vel_kmh = config.vel_kmh;
        
        % Calculate Doppler
        fD = (vel_kmh/3.6)/c * fc;
        
        % Determine mobility label
        if vel_kmh >= 80
            mobility = 'Very_High';
        else
            mobility = 'High';
        end
        
        fprintf('  Model %d/%d: %s @ %d km/h (fD≈%.1f Hz)\n', ...
                n_model, N_MODELS, model_name, vel_kmh, fD);
        
        % Generate channel object for this model
        rchan = ch_func.GenFadingChannel(model_name, fD, fs);
        
        % Calculate sample indices for this model
        sample_offset = (n_model - 1) * N_CH_PER_MODEL;
        
        % Loop over channel realizations
        for n_ch = 1:N_CH_PER_MODEL
            global_idx = sample_offset + n_ch;
            
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

            % -------- Store in COMBINED structures --------
            TX_Bits_Stream_Structure(:, global_idx)         = single(Bits_Stream_Coded);
            Received_Symbols_FFT_Structure(:, :, global_idx) = single(y_r(Kset,:));
            True_Channels_Structure(:, :, global_idx)       = single(hf(Kset,:));
            HLS_Structure(:, global_idx)                    = single(he_LS_Preamble(:));
            DPA_Structure(:, :, global_idx)                 = single(H_DPA);
            DPA_TA_Structure(:, :, global_idx)              = single(H_DPA_TA);
            STA_Structure(:, :, global_idx)                 = single(H_STA);
            TRFI_Structure(:, :, global_idx)                = single(H_TRFI);
            R_Symbols_Training_Structure(:, :, global_idx)  = single(y_r(Kset,:));
            
            % Store metadata
            Channel_Model_Labels(global_idx) = model_name;
        end
    end

    % -------- Save COMBINED file for this SNR ------------------------
    filename = sprintf('./MixedChannel_%s_%s_training_simulation_%d.mat', ...
                       snr_label, modu, EbN0dB(n_snr));
                       
    save(filename, ...
         'TX_Bits_Stream_Structure', ...
         'Received_Symbols_FFT_Structure', ...
         'True_Channels_Structure', ...
         'HLS_Structure', ...
         'DPA_Structure', ...
         'DPA_TA_Structure', ...
         'STA_Structure', ...
         'TRFI_Structure', ...
         'R_Symbols_Training_Structure', ...
         'Channel_Model_Labels', ...
         'Kset', 'ppositions', 'dpositions', ...
         'channel_configs', ...  % Save channel configuration info
         '-v7.3');
         
    fprintf('  Saved: %s (%.1f MB)\n', filename, dir(filename).bytes/1e6);
    toc;
end

fprintf('\n=== Done! ===\n');
fprintf('Generated %d files total\n', N_SNR);

end % function
