%% build_TCN_dataset_testing_individual.m
% TCN dataset builder for INDIVIDUAL CHANNEL testing data
% Handles files from gen_mixed_channel_testing.m with format:
%   <mobility>_<ChType>_<modu>_testing_simulation_<SNR>.mat
%
% forma:
%  - Inputs X: <scheme> channel estimates for ALL nSym symbols (52)
%  - Targets Y: TRUE channel for ALL nSym symbols (52 subcarrier)
%  - Format: [N x time x nSC] with nSC = 52, time = 2*nSym (real/imag interleaved)

clc; clear; close all; warning('off','all');

%% =================== USER SETTINGS ===================
mobility  = 'High';          % 'High' or 'Very_High'
ChType    = 'RTV_UC';             % 'VTV_EX', 'VTV_SDWW', 'RTV_SUS', etc.
modu      = '16QAM';
scheme    = 'DPA_TA';                % 'DPA','DPA_TA','TRFI','STA','LS', etc.

% File name pattern for individual channel testing data:
%   <mobility>_<ChType>_<modu>_testing_simulation_<SNR>.mat

%% =================== FIND FILES =======================
pat = sprintf('%s_%s_%s_testing_simulation_*.mat', mobility, ChType, modu);
lst = dir(pat);

% Also try short mobility tag (e.g., 'Very' instead of 'Very_High')
if isempty(lst)
    short_mob = regexprep(mobility, '_.*$', '');
    pat = sprintf('%s_%s_%s_testing_simulation_*.mat', short_mob, ChType, modu);
    lst = dir(pat);
end

assert(~isempty(lst), 'No files found for pattern: %s_%s_%s', mobility, ChType, modu);

% Sort by SNR suffix (ascending)
snr_vals = zeros(numel(lst), 1);
for k = 1:numel(lst)
    tok = regexp(lst(k).name, '_(\d+)\.mat$', 'tokens');
    assert(~isempty(tok), 'Cannot parse SNR from %s', lst(k).name);
    snr_vals(k) = str2double(tok{1}{1});
end
[snr_vals, order] = sort(snr_vals);
lst = lst(order);

fprintf('Found %d testing files for %s/%s/%s\n', numel(lst), mobility, ChType, modu);
fprintf('SNR list: '); fprintf('%d ', snr_vals); fprintf('\n\n');

%% ============ PEEK FIRST FILE TO LOCK SIZES ==========
peek = load(lst(1).name, 'True_Channels_Structure', [scheme '_Structure'], ...
            'Kset', 'dpositions', 'ppositions', 'Received_Symbols_FFT_Structure');

True_1    = peek.True_Channels_Structure;           % (Kon x nSym x N1)
Scheme_1  = peek.( [scheme '_Structure'] );         % (Kon x nSym x N1)
Kon_file  = size(True_1, 1);

% Subcarrier sets (fallbacks if not stored)
if isfield(peek, 'Kset'),       Kset       = peek.Kset(:);       else, Kset = (1:Kon_file).'; end
if isfield(peek, 'ppositions'), ppositions = peek.ppositions(:); else, ppositions = [7;21;32;46]; end
if isfield(peek, 'dpositions'), dpositions = peek.dpositions(:); else
    all_sc     = (1:numel(Kset)).';
    dpositions = setdiff(all_sc, ppositions, 'stable');
end

nSym_scheme = size(Scheme_1, 2);       % expected 50
nSym_true   = size(True_1,   2);       % expected 50

% Verify time alignment
assert(nSym_true == nSym_scheme, 'True has %d symbols, expected %d', nSym_true, nSym_scheme);

% Use ALL active subcarrier (52) for BOTH X and Y
used_idx = (1:numel(Kset)).';   % 52 active subcarrier
Kon_used = numel(used_idx);     % 52

% Time lengths: interleave Re/Im
nTime_X = 2 * nSym_scheme;      % 100 for 50 symbols
nTime_Y = 2 * nSym_scheme;      % 100

fprintf('Dataset configuration:\n');
fprintf('  nSym: %d\n', nSym_scheme);
fprintf('  Active subcarrier: %d (all)\n', Kon_used);
fprintf('  Time dimension: %d (real/imag interleaved)\n', nTime_X);
fprintf('  Data subcarrier: %d\n', numel(dpositions));
fprintf('  Pilot subcarrier: %d\n\n', numel(ppositions));

%% ============ PROCESS EACH SNR FILE =================
fprintf('Building testing datasets...\n\n');

for k = 1:numel(lst)
    fn = lst(k).name;
    snr_k = snr_vals(k);
    
    fprintf('SNR = %2d dB ... ', snr_k);
    tic;
    
    F = load(fn, 'True_Channels_Structure', [scheme '_Structure'], ...
             'Received_Symbols_FFT_Structure');

    True_blk   = F.True_Channels_Structure;           % (Kon x nSym x N)
    Scheme_blk = F.( [scheme '_Structure'] );         % (Kon x nSym x N)
    Yfft_blk   = F.Received_Symbols_FFT_Structure;    % (Kon x nSym x N)

    % Select ALL active subcarrier for BOTH X and Y (52 subcarrier)
    Scheme_blk = Scheme_blk(used_idx, :, :);          % (52 x nSym x N)
    True_blk   = True_blk(used_idx,   :, :);          % (52 x nSym x N)
    N = size(True_blk, 3);

    % ===== Build X (vectorized, 52 subcarrier) =====
    Sch_r = real(Scheme_blk);
    Sch_i = imag(Scheme_blk);
    Test_X_acc = zeros(Kon_used, nTime_X, N, 'single');
    Test_X_acc(:, 1:2:end, :) = single(Sch_r);
    Test_X_acc(:, 2:2:end, :) = single(Sch_i);

    % ===== Build Y (vectorized, 52 subcarrier) =====
    Tr_r = real(True_blk);
    Tr_i = imag(True_blk);
    Test_Y_acc = zeros(Kon_used, nTime_Y, N, 'single');
    Test_Y_acc(:, 1:2:end, :) = single(Tr_r);
    Test_Y_acc(:, 2:2:end, :) = single(Tr_i);

    % Observed symbols on ALL active subcarrier
    Y_active_obs = permute(Yfft_blk(used_idx, :, :), [3 2 1]);  % (N x nSym x 52)

    % Final layout: [N x Time x nSC] with nSC=52
    Test_X = permute(Test_X_acc, [3 2 1]);  % (N x 100 x 52)
    Test_Y = permute(Test_Y_acc, [3 2 1]);  % (N x 100 x 52)

    % Package & save (per SNR)
    TCN_Datasets.Test_X            = Test_X;
    TCN_Datasets.Test_Y            = Test_Y;
    TCN_Datasets.Y_DataSubCarriers = Y_active_obs;   % (N x nSym x 52)
    TCN_Datasets.DataSC_Idx        = dpositions;     % 1-based [48 indices]
    TCN_Datasets.PilotSC_Idx       = ppositions;     % 1-based [4 indices]

    % Save with consistent naming
    out_test = sprintf('./%s_%s_%s_%s_TCN_testing_dataset_%d.mat', ...
                      mobility, ChType, modu, scheme, snr_k);
    save(out_test, 'TCN_Datasets', '-v7.3');

    fprintf('Done (%.1fs, %.1f MB)\n', toc, dir(out_test).bytes/1e6);
    fprintf('  Shape: Test_X [%d x %d x %d], Test_Y [%d x %d x %d]\n', ...
            size(Test_X), size(Test_Y));
end

fprintf('\n========================================\n');
fprintf('ALL TESTING DATASETS SAVED\n');
fprintf('========================================\n');
fprintf('Channel: %s_%s\n', mobility, ChType);
fprintf('Modulation: %s\n', modu);
fprintf('Scheme: %s\n', scheme);
fprintf('Files created: %d (SNR 0-40 dB)\n', numel(lst));
fprintf('========================================\n\n');

fprintf('Done!\n');
