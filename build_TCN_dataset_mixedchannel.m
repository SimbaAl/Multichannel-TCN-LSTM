%% build_TCN_dataset_mixedchannel.m
% TCN dataset builder for MIXED CHANNEL training data
% Handles files from gen_mixed_channel_training.m with format:
%   MixedChannel_<SNRLabel>_<modu>_<mode>_simulation_<SNR>.mat
%
% format:
%  - Inputs X: <scheme> channel estimates for ALL nSym symbols,
%               with [real, imag] interleaved along time.
%  - Targets Y: TRUE channel for ALL nSym symbols (ALL 52 active),
%               with [real, imag] interleaved along time.
% Final layout: [N x time x nSC] with nSC = 52 (active subcarrier).

clc; clear; close all; warning('off','all');

%% =================== USER SETTINGS ===================
mode      = 'training';           % 'training' or 'testing'
modu      = '16QAM';
scheme    = 'DPA';                % 'DPA','DPA_TA','TRFI','STA','LS', etc.
snr_label = 'MixedSNR';           % 'MixedSNR' or 'HighSNR'

% If your generator's True includes a preamble symbol (nSym+1), set true
hasPreambleInTrue = false;        % true if True_Channels has (nSym+1) and needs dropping

% File name pattern for mixed channel data:
%   MixedChannel_<SNRLabel>_<modu>_<mode>_simulation_<SNR>.mat

%% =================== FIND FILES =======================
pat = sprintf('MixedChannel_%s_%s_%s_simulation_*.mat', snr_label, modu, mode);
lst = dir(pat);
assert(~isempty(lst), 'No files found for pattern: %s', pat);

% Sort by SNR suffix (ascending)
snr_vals = zeros(numel(lst),1);
for k = 1:numel(lst)
    tok = regexp(lst(k).name, '_(\d+)\.mat$', 'tokens');
    assert(~isempty(tok), 'Cannot parse SNR from %s', lst(k).name);
    snr_vals(k) = str2double(tok{1}{1});
end
[snr_vals, order] = sort(snr_vals);
lst = lst(order);

fprintf('Found %d "%s" files for MixedChannel/%s (SNRLabel=%s)\n', ...
        numel(lst), mode, modu, snr_label);
fprintf('SNR list: '); fprintf('%d ', snr_vals); fprintf('\n');

%% ============ PEEK FIRST FILE TO LOCK SIZES ==========
peek = load(lst(1).name, 'True_Channels_Structure', [scheme '_Structure'], ...
            'Kset', 'dpositions', 'ppositions', 'Received_Symbols_FFT_Structure', ...
            'Channel_Model_Labels');

True_1    = peek.True_Channels_Structure;           % (Kon x nSymTrue x N1)
Scheme_1  = peek.( [scheme '_Structure'] );         % (Kon x nSymScheme x N1)
Kon_file  = size(True_1,1);

% Check if Channel_Model_Labels exists (metadata from mixed channel generation)
if isfield(peek, 'Channel_Model_Labels')
    fprintf('Mixed channel metadata found: %d samples\n', length(peek.Channel_Model_Labels));
    % Display distribution of channel models in first file
    unique_models = unique(peek.Channel_Model_Labels);
    fprintf('Channel models in first file:\n');
    for i = 1:length(unique_models)
        count = sum(peek.Channel_Model_Labels == unique_models(i));
        fprintf('  %s: %d samples\n', unique_models(i), count);
    end
end

% Subcarrier sets (fallbacks if not stored)
if isfield(peek,'Kset'),       Kset       = peek.Kset(:);       else, Kset = (1:Kon_file).'; end
if isfield(peek,'ppositions'), ppositions = peek.ppositions(:); else, ppositions = [7;21;32;46]; end
if isfield(peek,'dpositions'), dpositions = peek.dpositions(:); else
    all_sc     = (1:numel(Kset)).';
    dpositions = setdiff(all_sc, ppositions, 'stable');
end

nSym_scheme = size(Scheme_1, 2);       % expected 50
nSym_true   = size(True_1,   2);       % 50 or 51 (if preamble)

% Verify time alignment (Scheme vs True)
if hasPreambleInTrue
    assert(nSym_true == nSym_scheme + 1, 'True has %d, expected nSym+1=%d', nSym_true, nSym_scheme+1);
    fprintf('Note: True includes preamble; will drop first symbol from True.\n');
else
    assert(nSym_true == nSym_scheme, 'True has %d, expected nSym=%d', nSym_true, nSym_scheme);
end

% Use ALL active subcarrier (52) for BOTH X and Y (paper-style)
used_idx = (1:numel(Kset)).';   % 52 active subcarrier
Kon_used = numel(used_idx);     % 52

% Time lengths: use scheme for ALL nSym; interleave Re/Im
nTime_X = 2 * nSym_scheme;
nTime_Y = 2 * nSym_scheme;
assert(nTime_X == nTime_Y, 'X and Y time lengths must match.');

%% ============ TRAINING: aggregate to ONE file =========
if strcmpi(mode, 'training')
    % Count total frames across all SNRs
    totalN = 0;
    all_labels = {};  % Collect all channel model labels
    
    for k = 1:numel(lst)
        S = load(lst(k).name, 'True_Channels_Structure', 'Channel_Model_Labels');
        totalN = totalN + size(S.True_Channels_Structure, 3);
        if isfield(S, 'Channel_Model_Labels')
            all_labels = [all_labels; cellstr(S.Channel_Model_Labels)];
        end
    end
    fprintf('Total frames across SNRs: %d\n', totalN);
    
    % Display overall distribution
    if ~isempty(all_labels)
        unique_models = unique(all_labels);
        fprintf('\nOverall channel model distribution:\n');
        for i = 1:length(unique_models)
            count = sum(strcmp(all_labels, unique_models{i}));
            fprintf('  %s: %d samples (%.1f%%)\n', unique_models{i}, count, 100*count/totalN);
        end
    end

    % Pre-allocate accumulators (both 52 x time)
    Train_X_acc = zeros(Kon_used, nTime_X, totalN, 'single');
    Train_Y_acc = zeros(Kon_used, nTime_Y, totalN, 'single');
    Channel_Labels_acc = strings(totalN, 1);  % Store labels

    write_at = 1;
    for k = 1:numel(lst)
        F = load(lst(k).name, 'True_Channels_Structure', [scheme '_Structure'], ...
                 'Received_Symbols_FFT_Structure', 'Channel_Model_Labels');
        True_blk   = F.True_Channels_Structure;        % (Kon x nSymTrue x Nk)
        Scheme_blk = F.( [scheme '_Structure'] );      % (Kon x nSymScheme x Nk)
        Nk         = size(True_blk, 3);

        if hasPreambleInTrue
            True_blk = True_blk(:, 2:end, :);          % drop preamble -> (Kon x nSym x Nk)
        end

        % Select ALL active subcarrier for BOTH X and Y
        Scheme_blk = Scheme_blk(used_idx, :, :);       % (52 x nSym x Nk)
        True_blk   = True_blk(used_idx,   :, :);       % (52 x nSym x Nk)

        % ===== Build X (vectorized, 52 subcarrier) =====
        Sch_r = real(Scheme_blk);
        Sch_i = imag(Scheme_blk);
        Sch_inter = zeros(Kon_used, nTime_X, Nk, 'single');
        Sch_inter(:,1:2:end,:) = single(Sch_r);
        Sch_inter(:,2:2:end,:) = single(Sch_i);

        % ===== Build Y (vectorized, 52 subcarrier) =====
        Tr_r = real(True_blk);
        Tr_i = imag(True_blk);
        Tr_inter = zeros(Kon_used, nTime_Y, Nk, 'single');
        Tr_inter(:,1:2:end,:) = single(Tr_r);
        Tr_inter(:,2:2:end,:) = single(Tr_i);

        % Write block
        idx = write_at:(write_at+Nk-1);
        Train_X_acc(:,:,idx) = Sch_inter;
        Train_Y_acc(:,:,idx) = Tr_inter;
        
        % Store channel labels if available
        if isfield(F, 'Channel_Model_Labels')
            Channel_Labels_acc(idx) = F.Channel_Model_Labels;
        end
        
        write_at = write_at + Nk;

        fprintf('  + %d frames from %s (SNR=%ddB)\n', Nk, lst(k).name, snr_vals(k));
    end
    assert(write_at-1 == totalN, 'Write count mismatch.')

    % Final layout: [N x Time x nSC] with nSC=52
    Train_X = permute(Train_X_acc, [3 2 1]);   % (N x 2*nSym x 52)
    Train_Y = permute(Train_Y_acc, [3 2 1]);   % (N x 2*nSym x 52)

    % Package & save
    TCN_Datasets.Train_X          = Train_X;
    TCN_Datasets.Train_Y          = Train_Y;
    TCN_Datasets.DataSC_Idx       = dpositions;    % 1-based (for evaluation)
    TCN_Datasets.PilotSC_Idx      = ppositions;    % 1-based
    TCN_Datasets.Channel_Labels   = Channel_Labels_acc;  % Metadata

    out_train = sprintf('./MixedChannel_%s_%s_%s_TCN_training_dataset.mat', ...
                        snr_label, modu, scheme);
    save(out_train, 'TCN_Datasets', '-v7.3');

    fprintf('\n=== TRAINING DATASET SAVED (MIXED CHANNEL, PAPER-STYLE, 52/52) ===\n');
    fprintf('  %s\n', out_train);
    fprintf('  Train_X: [%d x %d x %d], Train_Y: [%d x %d x %d]\n', size(Train_X), size(Train_Y));
    fprintf('  File size: %.1f MB\n', dir(out_train).bytes/1e6);

%% ============ TESTING: one file PER SNR =================
else
    fprintf('\n=== Building TESTING datasets ===\n');
    
    for k = 1:numel(lst)
        fn = lst(k).name;
        snr_k = snr_vals(k);
        F = load(fn, 'True_Channels_Structure', [scheme '_Structure'], ...
                 'Received_Symbols_FFT_Structure', 'Channel_Model_Labels');

        True_blk   = F.True_Channels_Structure;           % (Kon x nSymTrue x N)
        Scheme_blk = F.( [scheme '_Structure'] );         % (Kon x nSymScheme x N)
        Yfft_blk   = F.Received_Symbols_FFT_Structure;    % (Kon x nSym x N)

        if hasPreambleInTrue
            True_blk = True_blk(:, 2:end, :);             % drop preamble
            if size(Yfft_blk,2) == nSym_scheme+1
                Yfft_blk = Yfft_blk(:, 2:end, :);
            end
        end

        % Select ALL active subcarrier for BOTH X and Y
        Scheme_blk = Scheme_blk(used_idx, :, :);          % (52 x nSym x N)
        True_blk   = True_blk(used_idx,   :, :);          % (52 x nSym x N)
        N = size(True_blk, 3);
        
        % Display distribution for this SNR
        if isfield(F, 'Channel_Model_Labels')
            unique_models = unique(F.Channel_Model_Labels);
            fprintf('  SNR %ddB - Channel distribution:\n', snr_k);
            for i = 1:length(unique_models)
                count = sum(F.Channel_Model_Labels == unique_models(i));
                fprintf('    %s: %d samples\n', unique_models(i), count);
            end
        end

        % ===== Build X (vectorized, 52 subcarrier) =====
        Sch_r = real(Scheme_blk);
        Sch_i = imag(Scheme_blk);
        Test_X_acc = zeros(Kon_used, nTime_X, N, 'single');
        Test_X_acc(:,1:2:end,:) = single(Sch_r);
        Test_X_acc(:,2:2:end,:) = single(Sch_i);

        % ===== Build Y (vectorized, 52 subcarrier) =====
        Tr_r = real(True_blk);
        Tr_i = imag(True_blk);
        Test_Y_acc = zeros(Kon_used, nTime_Y, N, 'single');
        Test_Y_acc(:,1:2:end,:) = single(Tr_r);
        Test_Y_acc(:,2:2:end,:) = single(Tr_i);

        % Observed symbols on ALL active subcarrier
        Y_active_obs = permute(Yfft_blk(used_idx,:,:), [3 2 1]);

        % Final layout: [N x Time x nSC] with nSC=52
        Test_X = permute(Test_X_acc, [3 2 1]);
        Test_Y = permute(Test_Y_acc, [3 2 1]);

        % Package & save (per SNR)
        TCN_Datasets.Test_X            = Test_X;
        TCN_Datasets.Test_Y            = Test_Y;
        TCN_Datasets.Y_DataSubCarriers = Y_active_obs;   % (N x nSym x 52)
        TCN_Datasets.DataSC_Idx        = dpositions;     % 1-based
        TCN_Datasets.PilotSC_Idx       = ppositions;     % 1-based
        
        % Store channel labels if available
        if isfield(F, 'Channel_Model_Labels')
            TCN_Datasets.Channel_Labels = F.Channel_Model_Labels;
        end

        out_test = sprintf('./MixedChannel_%s_%s_%s_TCN_testing_dataset_%d.mat', ...
                          snr_label, modu, scheme, snr_k);
        save(out_test, 'TCN_Datasets', '-v7.3');

        fprintf('Saved testing dataset @ %ddB: %s (%.1f MB)\n', ...
                snr_k, out_test, dir(out_test).bytes/1e6);
        fprintf('  Test_X: [%d x %d x %d], Test_Y: [%d x %d x %d]\n', ...
                size(Test_X), size(Test_Y));
    end
    fprintf('\n=== ALL TESTING DATASETS SAVED (MIXED CHANNEL, PAPER-STYLE, 52/52) ===\n');
end

fprintf('\nDone!\n');
