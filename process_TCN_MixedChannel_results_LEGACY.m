function process_TCN_MixedChannel_results_LEGACY()
clc; clearvars; close all; warning('off','all');

%% ===================== USER CONFIG =====================
snr_label = 'MixedSNR';      % 'MixedSNR' or 'HighSNR' (training label)
ChType    = 'RTV_SUS';      % 'RTV_SUS', 'VTV_EX', 'VTV_SDWW', etc.
modu      = '16QAM';
scheme    = 'DPA';
tag       = 'TCN';           % Model tag from Python: 'TCN'

compare_classical = true;    % Include classical baselines (LS, DPA, etc.)

%% ===================== Determine Mobility =====================
if contains(ChType, 'RTV')
    mobility = 'High';
else
    mobility = 'Very_High';
end

fprintf('\n========================================\n');
fprintf('Processing Mixed Channel TCN Results [LEGACY DECODER]\n');
fprintf('========================================\n');
fprintf('Trained on: %s\n', snr_label);
fprintf('Tested on: %s (%s mobility)\n', ChType, mobility);
fprintf('Modulation: %s\n', modu);
fprintf('Scheme: %s\n', scheme);
fprintf('========================================\n\n');

%% ===================== System Parameters =====================
switch modu
    case 'QPSK',  nBitPerSym = 2;
    case '16QAM', nBitPerSym = 4;
    case '64QAM', nBitPerSym = 6;
    otherwise, error('Unsupported modulation');
end
M   = 2^nBitPerSym;
Pow = mean(abs(qammod(0:M-1, M)).^2);

nSym = 50;
Kon  = 52;
ppositions = [7, 21, 32, 46].';
dpositions = [1:6, 8:20, 22:31, 33:45, 47:52].';
nDSC = numel(dpositions);

% FEC chain
constlen = 7;
trellis  = poly2trellis(constlen, [171 133]);
tbl      = 34;
scramInit = 93;
Interleaver_Rows    = 16;
Interleaver_Columns = (nBitPerSym * nDSC * nSym) / Interleaver_Rows;

% SNR sweep
EbN0dB = (0:5:40).';
N_SNR  = numel(EbN0dB);

% Safe division helper
clipH = @(h) max(abs(h), 1e-8) .* exp(1j*angle(h + 1e-12));

% Allow short mobility prefix
short_mob = regexprep(mobility, '_.*$', '');

%% ===================== Load Random Permutation =====================
Random_permutation_Vector = [];
for n = 1:N_SNR
    sim_candidates = {
        sprintf('./%s_%s_%s_testing_simulation_%d.mat', mobility,  ChType, modu, EbN0dB(n));
        sprintf('./%s_%s_%s_testing_simulation_%d.mat', short_mob, ChType, modu, EbN0dB(n));
    };
    for k = 1:numel(sim_candidates)
        if exist(sim_candidates{k}, 'file')
            tmp = load(sim_candidates{k}, 'Random_permutation_Vector');
            if isfield(tmp, 'Random_permutation_Vector')
                Random_permutation_Vector = tmp.Random_permutation_Vector;
                fprintf('Loaded Random_permutation_Vector from: %s\n', sim_candidates{k});
                break;
            end
        end
    end
    if ~isempty(Random_permutation_Vector), break; end
end
assert(~isempty(Random_permutation_Vector), 'Missing Random_permutation_Vector.');

%% ===================== Accumulators =====================
Phf          = zeros(N_SNR, 1);
Nsamp        = zeros(N_SNR, 1);

Ber_Ideal    = zeros(N_SNR, 1);

Err_TCN_pre  = zeros(N_SNR, 1);
Ber_TCN_pre  = zeros(N_SNR, 1);
Err_TCN_post = zeros(N_SNR, 1);
Ber_TCN_post = zeros(N_SNR, 1);

if compare_classical
    Err_LS = zeros(N_SNR, 1);  Err_DPA = zeros(N_SNR, 1);
    Err_STA = zeros(N_SNR, 1); Err_TRFI = zeros(N_SNR, 1);
    Err_CDP = zeros(N_SNR, 1); Err_DPA_TA = zeros(N_SNR, 1);

    Ber_LS = zeros(N_SNR, 1);  Ber_DPA = zeros(N_SNR, 1);
    Ber_STA = zeros(N_SNR, 1); Ber_TRFI = zeros(N_SNR, 1);
    Ber_CDP = zeros(N_SNR, 1); Ber_DPA_TA = zeros(N_SNR, 1);
end

%% ===================== Main Loop =====================
for n_snr = 1:N_SNR
    SNRdB = EbN0dB(n_snr);
    fprintf('\nSNR = %2d dB ... ', SNRdB); tic;

    %% Load simulation file
    sim_candidates = {
        sprintf('./%s_%s_%s_testing_simulation_%d.mat', mobility,  ChType, modu, SNRdB);
        sprintf('./%s_%s_%s_testing_simulation_%d.mat', short_mob, ChType, modu, SNRdB);
    };
    sim_file = '';
    for k = 1:numel(sim_candidates)
        if exist(sim_candidates{k}, 'file')
            sim_file = sim_candidates{k};
            break;
        end
    end
    if isempty(sim_file)
        L = dir(sprintf('./*_%s_%s_testing_simulation_%d.mat', ChType, modu, SNRdB));
        if ~isempty(L)
            sim_file = fullfile(L(1).folder, L(1).name);
        end
    end
    if isempty(sim_file)
        warning('Missing testing_simulation for %d dB', SNRdB);
        continue;
    end

    S = load(sim_file);
    H_true_all = S.True_Channels_Structure;         % [52, 50, N]
    Y_rx_all   = S.Received_Symbols_FFT_Structure;  % [52, 50, N]
    TX_bits    = S.TX_Bits_Stream_Structure;        % [nInfoBits, N]
    N = size(H_true_all, 3);
    Nsamp(n_snr) = N;

    % Classical estimators
    if compare_classical
        getf = @(nm)(isfield(S, nm) && ~isempty(S.(nm)));
        if getf('HLS_Structure'),    HLS = S.HLS_Structure;          else, HLS = [];      end
        if getf('DPA_Structure'),    H_DPA = S.DPA_Structure;        else, H_DPA = [];    end
        if getf('DPA_TA_Structure'), H_DPA_TA = S.DPA_TA_Structure;  else, H_DPA_TA = []; end
        if getf('STA_Structure'),    H_STA = S.STA_Structure;        else, H_STA = [];    end
        if getf('TRFI_Structure'),   H_TRFI = S.TRFI_Structure;      else, H_TRFI = [];   end
        if getf('CDP_Structure'),    H_CDP = S.CDP_Structure;        else, H_CDP = [];    end
    end

    %% Load TCN results file (Mixed-channel model)
    result_candidates = {
        sprintf('./MixedChannel_%s_%s_%s_%s_%s_Results_%d.mat', snr_label, modu, scheme, tag, ChType, SNRdB);
        sprintf('./%s_%s_%s_%s_%s_Results_%d.mat', mobility, ChType, modu, scheme, tag, SNRdB);
    };
    res_file = '';
    for k = 1:numel(result_candidates)
        if exist(result_candidates{k}, 'file')
            res_file = result_candidates{k};
            break;
        end
    end
    if isempty(res_file)
        L = dir(sprintf('./MixedChannel_*_%s_%s_%s_%s_Results_%d.mat', modu, scheme, tag, ChType, SNRdB));
        if ~isempty(L)
            res_file = fullfile(L(1).folder, L(1).name);
        end
    end
    if isempty(res_file)
        warning('Missing TCN results for %d dB', SNRdB);
        continue;
    end

    T = load(res_file);

    % Candidates
    cand_pre  = {sprintf('%s_%s_pred_preDD_%d',  scheme, tag, SNRdB)};
    cand_post = {sprintf('%s_%s_pred_postDD_%d', scheme, tag, SNRdB)};

    H_pre = [];
    H_post_d = [];

    for c = 1:numel(cand_pre)
        if isfield(T, cand_pre{c})
            H_pre = T.(cand_pre{c});
            break;
        end
    end
    for c = 1:numel(cand_post)
        if isfield(T, cand_post{c})
            H_post_d = T.(cand_post{c});
            break;
        end
    end

    % Shape fixes
    if ~isempty(H_pre)
        if isequal(size(H_pre), [50, 52, N])
            H_pre = permute(H_pre, [3, 1, 2]);
        elseif isequal(size(H_pre), [52, 50, N])
            H_pre = permute(H_pre, [3, 2, 1]);
        end
        assert(isequal(size(H_pre), [N, 50, 52]), 'pre-DD must be [N,50,52], got %s', mat2str(size(H_pre)));
    end

    if ~isempty(H_post_d)
        if isequal(size(H_post_d), [48, 50, N])
            H_post_d = permute(H_post_d, [3, 2, 1]);
        elseif isequal(size(H_post_d), [50, 48, N])
            H_post_d = permute(H_post_d, [3, 1, 2]);
        end
        assert(isequal(size(H_post_d), [N, 50, 48]), 'post-DD must be [N,50,48], got %s', mat2str(size(H_post_d)));
    end

    %% Per-frame processing
    for u = 1:N
        H_true   = H_true_all(:, :, u);        % [52, 50]
        Y_rx     = Y_rx_all(:, :, u);          % [52, 50]
        H_true_d = H_true(dpositions, :);      % [48, 50]
        Y_rx_d   = Y_rx(dpositions, :);        % [48, 50]

        % Channel power
        Phf(n_snr) = Phf(n_snr) + mean(sum(abs(H_true_d).^2));

        %% Ideal BER (perfect channel)
        Y_eq_I = Y_rx_d ./ clipH(H_true_d);
        Bits_I_3D = demap_qam_bits(Y_eq_I, M, Pow, nBitPerSym);
        Bits_I = Bits_I_3D(:);

        % LEGACY DECODER
        Bits_I_dec = decode_bits_legacy(Bits_I, Random_permutation_Vector, ...
                                        Interleaver_Columns, Interleaver_Rows, ...
                                        trellis, tbl, scramInit);
        Ber_Ideal(n_snr) = Ber_Ideal(n_snr) + biterr(Bits_I_dec, TX_bits(:, u));

        %% TCN pre-DD
        if ~isempty(H_pre)
            H_pre_frame = squeeze(H_pre(u, :, :)).';   % [52,50]
            H_pre_d = H_pre_frame(dpositions, :);      % [48,50]

            Err_TCN_pre(n_snr) = Err_TCN_pre(n_snr) + mean(sum(abs(H_pre_d - H_true_d).^2));

            Y_eq_pre = Y_rx_d ./ clipH(H_pre_d);
            Bits_pre_3D = demap_qam_bits(Y_eq_pre, M, Pow, nBitPerSym);
            Bits_pre = Bits_pre_3D(:);

            Bits_pre_dec = decode_bits_legacy(Bits_pre, Random_permutation_Vector, ...
                                              Interleaver_Columns, Interleaver_Rows, ...
                                              trellis, tbl, scramInit);
            Ber_TCN_pre(n_snr) = Ber_TCN_pre(n_snr) + biterr(Bits_pre_dec, TX_bits(:, u));
        end

        %% TCN post-DD
        if ~isempty(H_post_d)
            H_post = squeeze(H_post_d(u, :, :)).';     % [48,50]

            Err_TCN_post(n_snr) = Err_TCN_post(n_snr) + mean(sum(abs(H_post - H_true_d).^2));

            Y_eq_post = Y_rx_d ./ clipH(H_post);
            Bits_post_3D = demap_qam_bits(Y_eq_post, M, Pow, nBitPerSym);
            Bits_post = Bits_post_3D(:);

            Bits_post_dec = decode_bits_legacy(Bits_post, Random_permutation_Vector, ...
                                               Interleaver_Columns, Interleaver_Rows, ...
                                               trellis, tbl, scramInit);
            Ber_TCN_post(n_snr) = Ber_TCN_post(n_snr) + biterr(Bits_post_dec, TX_bits(:, u));
        end

        %% Classical baselines (also using LEGACY decoder for consistency)
        if compare_classical
            % LS
            if ~isempty(HLS)
                H_LS_d = repmat(HLS(dpositions, u), 1, nSym);
                Err_LS(n_snr) = Err_LS(n_snr) + mean(sum(abs(H_LS_d - H_true_d).^2));

                Bits_LS_3D = demap_qam_bits(Y_rx_d ./ clipH(H_LS_d), M, Pow, nBitPerSym);
                Bits_LS = Bits_LS_3D(:);
                Ber_LS(n_snr) = Ber_LS(n_snr) + biterr(...
                    decode_bits_legacy(Bits_LS, Random_permutation_Vector, ...
                                       Interleaver_Columns, Interleaver_Rows, ...
                                       trellis, tbl, scramInit), ...
                    TX_bits(:, u));
            end

            % DPA
            if ~isempty(H_DPA)
                H_DPA_d = H_DPA(dpositions, :, u);
                Err_DPA(n_snr) = Err_DPA(n_snr) + mean(sum(abs(H_DPA_d - H_true_d).^2));

                Bits_DPA_3D = demap_qam_bits(Y_rx_d ./ clipH(H_DPA_d), M, Pow, nBitPerSym);
                Bits_DPA = Bits_DPA_3D(:);
                Ber_DPA(n_snr) = Ber_DPA(n_snr) + biterr(...
                    decode_bits_legacy(Bits_DPA, Random_permutation_Vector, ...
                                       Interleaver_Columns, Interleaver_Rows, ...
                                       trellis, tbl, scramInit), ...
                    TX_bits(:, u));
            end

            % DPA-TA
            if ~isempty(H_DPA_TA)
                H_DPA_TA_d = H_DPA_TA(dpositions, :, u);
                Err_DPA_TA(n_snr) = Err_DPA_TA(n_snr) + mean(sum(abs(H_DPA_TA_d - H_true_d).^2));

                Bits_DPA_TA_3D = demap_qam_bits(Y_rx_d ./ clipH(H_DPA_TA_d), M, Pow, nBitPerSym);
                Bits_DPA_TA = Bits_DPA_TA_3D(:);
                Ber_DPA_TA(n_snr) = Ber_DPA_TA(n_snr) + biterr(...
                    decode_bits_legacy(Bits_DPA_TA, Random_permutation_Vector, ...
                                       Interleaver_Columns, Interleaver_Rows, ...
                                       trellis, tbl, scramInit), ...
                    TX_bits(:, u));
            end

            % STA
            if ~isempty(H_STA)
                H_STA_d = H_STA(dpositions, :, u);
                Err_STA(n_snr) = Err_STA(n_snr) + mean(sum(abs(H_STA_d - H_true_d).^2));

                Bits_STA_3D = demap_qam_bits(Y_rx_d ./ clipH(H_STA_d), M, Pow, nBitPerSym);
                Bits_STA = Bits_STA_3D(:);
                Ber_STA(n_snr) = Ber_STA(n_snr) + biterr(...
                    decode_bits_legacy(Bits_STA, Random_permutation_Vector, ...
                                       Interleaver_Columns, Interleaver_Rows, ...
                                       trellis, tbl, scramInit), ...
                    TX_bits(:, u));
            end

            % TRFI
            if ~isempty(H_TRFI)
                H_TRFI_d = H_TRFI(dpositions, :, u);
                Err_TRFI(n_snr) = Err_TRFI(n_snr) + mean(sum(abs(H_TRFI_d - H_true_d).^2));

                Bits_TRFI_3D = demap_qam_bits(Y_rx_d ./ clipH(H_TRFI_d), M, Pow, nBitPerSym);
                Bits_TRFI = Bits_TRFI_3D(:);
                Ber_TRFI(n_snr) = Ber_TRFI(n_snr) + biterr(...
                    decode_bits_legacy(Bits_TRFI, Random_permutation_Vector, ...
                                       Interleaver_Columns, Interleaver_Rows, ...
                                       trellis, tbl, scramInit), ...
                    TX_bits(:, u));
            end

            % CDP
            if ~isempty(H_CDP)
                H_CDP_d = H_CDP(:, :, u);  % [48,50]
                Err_CDP(n_snr) = Err_CDP(n_snr) + mean(sum(abs(H_CDP_d - H_true_d).^2));

                Bits_CDP_3D = demap_qam_bits(Y_rx_d ./ clipH(H_CDP_d), M, Pow, nBitPerSym);
                Bits_CDP = Bits_CDP_3D(:);
                Ber_CDP(n_snr) = Ber_CDP(n_snr) + biterr(...
                    decode_bits_legacy(Bits_CDP, Random_permutation_Vector, ...
                                       Interleaver_Columns, Interleaver_Rows, ...
                                       trellis, tbl, scramInit), ...
                    TX_bits(:, u));
            end
        end
    end

    fprintf('Done (%.2fs)\n', toc);
end

%% ===================== Normalize =====================
Phf = Phf ./ max(Nsamp, 1);
bits_total = Nsamp .* nSym .* nDSC .* nBitPerSym;

BER_Ideal    = Ber_Ideal    ./ max(bits_total, 1);
BER_TCN_pre  = Ber_TCN_pre  ./ max(bits_total, 1);
BER_TCN_post = Ber_TCN_post ./ max(bits_total, 1);

NMSE_TCN_pre  = Err_TCN_pre  ./ max(Nsamp .* Phf, 1);
NMSE_TCN_post = Err_TCN_post ./ max(Nsamp .* Phf, 1);

if compare_classical
    NMSE_LS     = Err_LS     ./ max(Nsamp .* Phf, 1);
    NMSE_DPA    = Err_DPA    ./ max(Nsamp .* Phf, 1);
    NMSE_DPA_TA = Err_DPA_TA ./ max(Nsamp .* Phf, 1);
    NMSE_STA    = Err_STA    ./ max(Nsamp .* Phf, 1);
    NMSE_TRFI   = Err_TRFI   ./ max(Nsamp .* Phf, 1);
    NMSE_CDP    = Err_CDP    ./ max(Nsamp .* Phf, 1);

    BER_LS     = Ber_LS     ./ max(bits_total, 1);
    BER_DPA    = Ber_DPA    ./ max(bits_total, 1);
    BER_DPA_TA = Ber_DPA_TA ./ max(bits_total, 1);
    BER_STA    = Ber_STA    ./ max(bits_total, 1);
    BER_TRFI   = Ber_TRFI   ./ max(bits_total, 1);
    BER_CDP    = Ber_CDP    ./ max(bits_total, 1);
end

%% ===================== Save Results =====================
out_base = sprintf('./MixedChannel_%s_%s_%s_%s_%s_legacy_results', ...
                   snr_label, ChType, modu, scheme, tag);

save([out_base '.mat'], ...
     'EbN0dB', 'BER_Ideal', 'BER_TCN_pre', 'BER_TCN_post', ...
     'NMSE_TCN_pre', 'NMSE_TCN_post', '-v7.3');

if compare_classical
    save([out_base '_comparison.mat'], ...
         'EbN0dB', 'BER_Ideal', 'BER_TCN_pre', 'BER_TCN_post', ...
         'NMSE_TCN_pre', 'NMSE_TCN_post', ...
         'NMSE_LS', 'NMSE_DPA', 'NMSE_DPA_TA', 'NMSE_STA', 'NMSE_TRFI', 'NMSE_CDP', ...
         'BER_LS', 'BER_DPA', 'BER_DPA_TA', 'BER_STA', 'BER_TRFI', 'BER_CDP', ...
         '-v7.3');
end

%% ===================== Print Summary =====================
fprintf('\n========================================\n');
fprintf('BER Summary [LEGACY DECODER]\n');
fprintf('========================================\n');
disp(table(EbN0dB, BER_Ideal, BER_TCN_pre, BER_TCN_post, ...
          'VariableNames', {'SNR_dB', 'Ideal', 'TCN_Pre', 'TCN_Post'}));

fprintf('\n========================================\n');
fprintf('NMSE Summary (Linear Scale)\n');
fprintf('========================================\n');
disp(table(EbN0dB, NMSE_TCN_pre, NMSE_TCN_post, ...
          'VariableNames', {'SNR_dB', 'TCN_Pre', 'TCN_Post'}));

%% ===================== PLOTS =====================
plot_basename = sprintf('MixedChannel_%s_%s_%s_%s_%s_legacy', ...
                       snr_label, ChType, modu, scheme, tag);

% NMSE
fig1 = figure('Position', [100, 100, 1000, 650]);
semilogy(EbN0dB, NMSE_TCN_pre,  'b--v','LineWidth',2.5,'MarkerSize',8,...
         'MarkerFaceColor','b','DisplayName','TCN Pre-DD'); hold on;
semilogy(EbN0dB, NMSE_TCN_post, 'b-o', 'LineWidth',2.8,'MarkerSize',9,...
         'MarkerFaceColor','b','DisplayName','TCN Post-DD');
if compare_classical
    semilogy(EbN0dB, NMSE_LS,     'k--s','LineWidth',2,'DisplayName','LS');
    semilogy(EbN0dB, NMSE_DPA,    'r-^', 'LineWidth',2,'MarkerFaceColor','r','DisplayName','DPA');
    semilogy(EbN0dB, NMSE_DPA_TA, 'g-d', 'LineWidth',2,'MarkerFaceColor','g','DisplayName','DPA-TA');
    semilogy(EbN0dB, NMSE_STA,    'm->', 'LineWidth',2,'MarkerFaceColor','m','DisplayName','STA');
    semilogy(EbN0dB, NMSE_TRFI,   'c-p', 'LineWidth',2,'DisplayName','TRFI');
    semilogy(EbN0dB, NMSE_CDP,    'Color',[1 0.6 0],'Marker','h','LineWidth',2,'DisplayName','CDP');
end
grid on; box on;
xlabel('SNR (dB)'); ylabel('NMSE (Linear)');
title(sprintf('NMSE [LEGACY DECODER]: Trained %s, Tested %s (%s)', ...
      snr_label, ChType, modu), 'Interpreter','none');
xlim([min(EbN0dB), max(EbN0dB)]); ylim([1e-3,1e2]);
legend('Location','northeast');
saveas(fig1, [plot_basename '_NMSE.png']);
fprintf('\nSaved: %s_NMSE.png\n', plot_basename);

% BER
fig2 = figure('Position', [100, 100, 1000, 650]);
semilogy(EbN0dB, BER_Ideal,     'k-o','LineWidth',2.5,'MarkerSize',8,...
         'MarkerFaceColor','k','DisplayName','Perfect Channel'); hold on;
semilogy(EbN0dB, BER_TCN_pre,  'b--v','LineWidth',2.5,'MarkerSize',8,...
         'MarkerFaceColor','b','DisplayName','TCN Pre-DD');
semilogy(EbN0dB, BER_TCN_post, 'b-o','LineWidth',2.8,'MarkerSize',9,...
         'MarkerFaceColor','b','DisplayName','TCN Post-DD');
if compare_classical
    semilogy(EbN0dB, BER_LS,     'r--s','LineWidth',2,'DisplayName','LS');
    semilogy(EbN0dB, BER_DPA,    'r-^', 'LineWidth',2,'MarkerFaceColor','r','DisplayName','DPA');
    semilogy(EbN0dB, BER_DPA_TA, 'g-d', 'LineWidth',2,'MarkerFaceColor','g','DisplayName','DPA-TA');
    semilogy(EbN0dB, BER_STA,    'm->', 'LineWidth',2,'MarkerFaceColor','m','DisplayName','STA');
    semilogy(EbN0dB, BER_TRFI,   'c-p', 'LineWidth',2,'DisplayName','TRFI');
    semilogy(EbN0dB, BER_CDP,    'Color',[1 0.6 0],'Marker','h','LineWidth',2,'DisplayName','CDP');
end
grid on; box on;
xlabel('SNR (dB)'); ylabel('BER');
title(sprintf('BER [LEGACY DECODER]: Trained %s, Tested %s (%s)', ...
      snr_label, ChType, modu), 'Interpreter','none');
xlim([min(EbN0dB), max(EbN0dB)]); ylim([1e-5,1e0]);
legend('Location','southwest');
saveas(fig2, [plot_basename '_BER.png']);
fprintf('Saved: %s_BER.png\n', plot_basename);

fprintf('\n========================================\n');
fprintf('Processing Complete [LEGACY DECODER]!\n');
fprintf('Saved: %s.mat\n', out_base);
if compare_classical
    fprintf('Saved: %s_comparison.mat\n', out_base);
end
fprintf('========================================\n\n');

end

%% ===================== Helper: Demap to Bits =====================
function Bits_3D = demap_qam_bits(Z, M, Pow, nBitPerSym)
    idx = qamdemod(sqrt(Pow) * Z, M, 'gray', 'OutputType', 'integer');
    sz = size(idx);
    idx = idx(:);
    b_flat = de2bi(idx, nBitPerSym, 'right-msb');
    Bits_3D = reshape(b_flat, [sz(1), sz(2), nBitPerSym]);
end

%% ===================== Helper: LEGACY Decode (LSTM-style) =====================
function decoded_bits = decode_bits_legacy(bits_vector, perm_vector, ...
                                           n_cols, n_rows, trellis, tbl, scramInit)
% Legacy behavior from original LSTM / TCN code:
% - deintrlv with perm_vector
% - transpose
% - matintrlv with (n_cols, n_rows)  [NOTE: NOT matdeintrlv, and dims swapped vs TX!]
% - vitdec
% - wlanScramble

    deinterleaved = deintrlv(bits_vector, perm_vector);  % undo intrlv
    deinterleaved = deinterleaved.';                     % row

    matrix_deinterleaved = matintrlv(deinterleaved, n_cols, n_rows);
    % (This is deliberately the "legacy" choice.)

    decoded = vitdec(matrix_deinterleaved.', trellis, tbl, 'trunc', 'hard');
    decoded_bits = wlanScramble(decoded, scramInit);
end
