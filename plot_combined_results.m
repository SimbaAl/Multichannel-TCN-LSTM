function plot_combined_results()
% Unified plotting script for TCN and LSTM-DPA-TA comparison
% Plots BER and NMSE results on separate figures with consistent styling.

clc; close all; warning('off','all');

%% ===================== USER CONFIG =====================
% TCN Configuration
tcn_snr_label = 'MixedSNR';      % 'MixedSNR' or 'HighSNR'
tcn_ChType    = 'VTV_SDWW';
tcn_modu      = '16QAM';
tcn_scheme    = 'DPA';
tcn_tag       = 'TCN';

% LSTM Configuration
lstm_METHOD_TAG    = 'LSTM-MixedChannel';
lstm_training_type = 'HighSNR';  % 'HighSNR' or 'MixedSNR'
lstm_mobility      = 'Very_High';
lstm_ChType        = 'VTV_SDWW';

% Plotting options
plot_classical = true;           % Include classical estimators
save_plots     = true;           % Save plots to files

%% ===================== STYLE (FROM REFERENCE SCRIPT) =====================
LW = 2.3;                % line width
MS = 7;                  % marker size
AX_FONT = 11;

set_x = @(ax, x_min, x_max) set(ax, ...
    'XLim',[x_min x_max], ...
    'XTick', x_min:5:x_max);

% Colors from reference script
DPA_COLOR = [0.55 0 0.55];   % DPA purple
COLOR_LS  = [0 0.5 0.5];     % teal
COLOR_CDP = [1 0.5 0];       % orange
COLOR_TRFI = [0 1 0];        % green
COLOR_STA = [1 0 1];         % magenta

% Deep Learning methods
COLOR_TCN_DPA     = [0 0 1];   % blue (keeping original)
COLOR_TCN_DPA_TA  = [1 0 0];   % red (for TCN pre)
COLOR_LSTM        = [0.50 0.50 0.00];  % olive

% Perfect Channel
STYLE_PERF = {'-','Color','k','LineWidth',LW,'Marker','o','MarkerSize',MS, ...
              'DisplayName','Perfect Channel'};

% TCN-DPA: BLUE dotted with square (keeping original as requested)
STYLE_TCN_DPA = {':','Color',COLOR_TCN_DPA,'LineWidth',LW,'Marker','s', ...
                 'MarkerFaceColor','none','MarkerSize',MS};

% TCN (pre): RED dotted with diamond (TCN-DPA-TA style from reference)
STYLE_TCN_PRE = {':','Color',COLOR_TCN_DPA_TA,'LineWidth',LW,'Marker','d', ...
                 'MarkerFaceColor','none','MarkerSize',MS};

% LSTM-DPA-TA: keeping olive dotted '<' as specified
STYLE_LSTM = {':','Color',COLOR_LSTM,'LineWidth',LW,'Marker','<', ...
              'MarkerSize',MS,'MarkerFaceColor','none'};

% Classical estimators - SOLID lines with reference colors
STYLE_LS = {'-','Color',COLOR_LS,'LineWidth',LW,'Marker','s', ...
            'MarkerSize',MS,'MarkerFaceColor','none'};

STYLE_DPA = {'-','Color',DPA_COLOR,'LineWidth',LW,'Marker','o', ...
             'MarkerSize',MS,'MarkerFaceColor','none'};

STYLE_DPA_TA = {'-','Color','c','LineWidth',LW,'Marker','d', ...
                'MarkerSize',MS,'MarkerFaceColor','none'};

STYLE_STA = {'-','Color',COLOR_STA,'LineWidth',LW,'Marker','v', ...
             'MarkerSize',MS,'MarkerFaceColor','none'};

STYLE_TRFI = {'-','Color',COLOR_TRFI,'LineWidth',LW,'Marker','s', ...
              'MarkerSize',MS,'MarkerFaceColor','none'};

STYLE_CDP = {'-','Color',COLOR_CDP,'LineWidth',LW,'Marker','h', ...
             'MarkerSize',MS,'MarkerFaceColor','none'};

%% ===================== Load TCN Results =====================
fprintf('========================================\n');
fprintf('Loading TCN Results\n');
fprintf('========================================\n');

tcn_base = sprintf('./MixedChannel_%s_%s_%s_%s_%s_results', ...
                   tcn_snr_label, tcn_ChType, tcn_modu, tcn_scheme, tcn_tag);

tcn_main_file = [tcn_base '.mat'];
tcn_comp_file = [tcn_base '_comparison.mat'];

has_tcn = false;
has_classical = false;

if exist(tcn_main_file, 'file')
    tcn_data = load(tcn_main_file);
    has_tcn = true;
    fprintf('✓ Loaded TCN main results\n');
    
    if plot_classical && exist(tcn_comp_file, 'file')
        tcn_comp = load(tcn_comp_file);
        has_classical = true;
        fprintf('✓ Loaded TCN classical comparison\n');
    else
        fprintf('  Classical comparison not available\n');
    end
else
    warning('TCN results file not found: %s', tcn_main_file);
end

%% ===================== Load LSTM-DPA-TA Results =====================
fprintf('\n========================================\n');
fprintf('Loading LSTM-DPA-TA Results\n');
fprintf('========================================\n');

lstm_filename = sprintf('metrics_%s_%s_%s_%s.mat', ...
    lstm_METHOD_TAG, lstm_training_type, lstm_mobility, lstm_ChType);

has_lstm = false;
if exist(lstm_filename, 'file')
    lstm_data = load(lstm_filename);
    has_lstm = true;
    fprintf('✓ Loaded LSTM-DPA-TA results\n');
else
    warning('LSTM-DPA-TA results file not found: %s', lstm_filename);
end

%% ===================== Check Data Availability =====================
if ~has_tcn && ~has_lstm
    error('No data available to plot!');
end

fprintf('\n========================================\n');
fprintf('Data Status\n');
fprintf('========================================\n');

%% ===================== Common SNR Range =====================
all_snr = [];
if has_tcn && isfield(tcn_data,'EbN0dB')
    all_snr = [all_snr; tcn_data.EbN0dB(:)];
end
if has_lstm && isfield(lstm_data,'EbN0dB')
    all_snr = [all_snr; lstm_data.EbN0dB(:)];
end

x_min = min(all_snr);
x_max = max(all_snr);

%% ===================== PLOT 1: BER Comparison =====================
fprintf('Creating BER plot...\n');

fig_ber = figure('Color','w','Position',[100,100,900,600]);
ax1 = gca;
hold(ax1,'on'); 
grid(ax1,'on'); 
box(ax1,'on');
set(ax1,'YScale','log','FontSize',AX_FONT);

% Store plot handles for legend ordering
h_plots = [];
labels = {};

% Perfect Channel
if has_tcn && isfield(tcn_data,'BER_Ideal')
    h = semilogy(tcn_data.EbN0dB, tcn_data.BER_Ideal, STYLE_PERF{:});
    h_plots = [h_plots; h];
    labels = [labels; 'Perfect Channel'];
end

% Classical estimators (in order: LS, DPA, STA, TRFI, DPA-TA, CDP)
if has_tcn && plot_classical && has_classical
    if isfield(tcn_comp,'BER_LS')
        h = semilogy(tcn_data.EbN0dB, tcn_comp.BER_LS, STYLE_LS{:});
        h_plots = [h_plots; h];
        labels = [labels; 'LS'];
    end
    if isfield(tcn_comp,'BER_DPA')
        h = semilogy(tcn_data.EbN0dB, tcn_comp.BER_DPA, STYLE_DPA{:});
        h_plots = [h_plots; h];
        labels = [labels; 'DPA'];
    end
    if isfield(tcn_comp,'BER_STA')
        h = semilogy(tcn_data.EbN0dB, tcn_comp.BER_STA, STYLE_STA{:});
        h_plots = [h_plots; h];
        labels = [labels; 'STA'];
    end
    if isfield(tcn_comp,'BER_TRFI')
        h = semilogy(tcn_data.EbN0dB, tcn_comp.BER_TRFI, STYLE_TRFI{:});
        h_plots = [h_plots; h];
        labels = [labels; 'TRFI'];
    end
    if isfield(tcn_comp,'BER_DPA_TA')
        h = semilogy(tcn_data.EbN0dB, tcn_comp.BER_DPA_TA, STYLE_DPA_TA{:});
        h_plots = [h_plots; h];
        labels = [labels; 'DPA-TA'];
    end
    if isfield(tcn_comp,'BER_CDP')
        h = semilogy(tcn_data.EbN0dB, tcn_comp.BER_CDP, STYLE_CDP{:});
        h_plots = [h_plots; h];
        labels = [labels; 'CDP'];
    end
end

% LSTM-DPA-TA
if has_lstm && isfield(lstm_data,'BER_scheme_LSTM')
    h = semilogy(lstm_data.EbN0dB, lstm_data.BER_scheme_LSTM, STYLE_LSTM{:});
    h_plots = [h_plots; h];
    labels = [labels; 'LSTM-DPA-TA'];
end

% TCN-DPA (post)
if has_tcn && isfield(tcn_data,'BER_TCN_post')
    h = semilogy(tcn_data.EbN0dB, tcn_data.BER_TCN_post, STYLE_TCN_DPA{:});
    h_plots = [h_plots; h];
    labels = [labels; 'TCN-DPA'];
end

% TCN (pre)
if has_tcn && isfield(tcn_data,'BER_TCN_pre')
    h = semilogy(tcn_data.EbN0dB, tcn_data.BER_TCN_pre, STYLE_TCN_PRE{:});
    h_plots = [h_plots; h];
    labels = [labels; 'TCN'];
end

xlabel('SNR (dB)');
ylabel('BER');
set_x(ax1, x_min, x_max);
ylim([1e-7, 1e0]);

legend(h_plots, labels, 'Location','southwest');

if save_plots
    ber_filename = sprintf('Combined_BER_%s_vs_%s_%s.png', ...
        tcn_snr_label, lstm_training_type, tcn_ChType);
    saveas(fig_ber, ber_filename);
    fprintf('✓ Saved BER plot: %s\n', ber_filename);
end

%% ===================== PLOT 2: NMSE Comparison =====================
fprintf('Creating NMSE plot...\n');

fig_nmse = figure('Color','w','Position',[150,150,900,600]);
ax2 = gca;
hold(ax2,'on'); 
grid(ax2,'on'); 
box(ax2,'on');
set(ax2,'YScale','log','FontSize',AX_FONT);

% Store plot handles for legend ordering
h_plots2 = [];
labels2 = {};

% Classical estimators (in order: LS, DPA, STA, TRFI, DPA-TA, CDP)
if has_tcn && plot_classical && has_classical
    if isfield(tcn_comp,'NMSE_LS')
        h = semilogy(tcn_data.EbN0dB, tcn_comp.NMSE_LS, STYLE_LS{:});
        h_plots2 = [h_plots2; h];
        labels2 = [labels2; 'LS'];
    end
    if isfield(tcn_comp,'NMSE_DPA')
        h = semilogy(tcn_data.EbN0dB, tcn_comp.NMSE_DPA, STYLE_DPA{:});
        h_plots2 = [h_plots2; h];
        labels2 = [labels2; 'DPA'];
    end
    if isfield(tcn_comp,'NMSE_STA')
        h = semilogy(tcn_data.EbN0dB, tcn_comp.NMSE_STA, STYLE_STA{:});
        h_plots2 = [h_plots2; h];
        labels2 = [labels2; 'STA'];
    end
    if isfield(tcn_comp,'NMSE_TRFI')
        h = semilogy(tcn_data.EbN0dB, tcn_comp.NMSE_TRFI, STYLE_TRFI{:});
        h_plots2 = [h_plots2; h];
        labels2 = [labels2; 'TRFI'];
    end
    if isfield(tcn_comp,'NMSE_DPA_TA')
        h = semilogy(tcn_data.EbN0dB, tcn_comp.NMSE_DPA_TA, STYLE_DPA_TA{:});
        h_plots2 = [h_plots2; h];
        labels2 = [labels2; 'DPA-TA'];
    end
    if isfield(tcn_comp,'NMSE_CDP')
        h = semilogy(tcn_data.EbN0dB, tcn_comp.NMSE_CDP, STYLE_CDP{:});
        h_plots2 = [h_plots2; h];
        labels2 = [labels2; 'CDP'];
    end
end

% LSTM-DPA-TA NMSE
if has_lstm && isfield(lstm_data,'ERR_scheme_LSTM')
    h = semilogy(lstm_data.EbN0dB, lstm_data.ERR_scheme_LSTM, STYLE_LSTM{:});
    h_plots2 = [h_plots2; h];
    labels2 = [labels2; 'LSTM-DPA-TA'];
end

% TCN-DPA NMSE
if has_tcn && isfield(tcn_data,'NMSE_TCN_post')
    h = semilogy(tcn_data.EbN0dB, tcn_data.NMSE_TCN_post, STYLE_TCN_DPA{:});
    h_plots2 = [h_plots2; h];
    labels2 = [labels2; 'TCN-DPA'];
end

% TCN NMSE (pre)
if has_tcn && isfield(tcn_data,'NMSE_TCN_pre')
    h = semilogy(tcn_data.EbN0dB, tcn_data.NMSE_TCN_pre, STYLE_TCN_PRE{:});
    h_plots2 = [h_plots2; h];
    labels2 = [labels2; 'TCN'];
end

xlabel('SNR (dB)');
ylabel('NMSE');
set_x(ax2, x_min, x_max);
ylim([1e-5, 1e2]);  % reverted to original

legend(h_plots2, labels2, 'Location','northeast');  % reverted to original

if save_plots
    nmse_filename = sprintf('Combined_NMSE_%s_vs_%s_%s.png', ...
        tcn_snr_label, lstm_training_type, tcn_ChType);
    saveas(fig_nmse, nmse_filename);
    fprintf('✓ Saved NMSE plot: %s\n', nmse_filename);
end

%% ===================== SUMMARY STATS =====================
fprintf('\n========================================\n');
fprintf('SUMMARY STATISTICS\n');
fprintf('========================================\n');

if has_tcn
    fprintf('\nTCN Results (Trained: %s, Tested: %s):\n', tcn_snr_label, tcn_ChType);
    if isfield(tcn_data,'BER_TCN_post')
        fprintf('  BER (TCN-DPA) @ max SNR: %.4e\n', tcn_data.BER_TCN_post(end));
    end
    if isfield(tcn_data,'NMSE_TCN_post')
        fprintf('  NMSE (TCN-DPA) @ max SNR: %.4e\n', tcn_data.NMSE_TCN_post(end));
    end
end

if has_lstm
    fprintf('\nLSTM-DPA-TA Results (Trained: %s, Tested: %s):\n', lstm_training_type, lstm_ChType);
    if isfield(lstm_data,'BER_scheme_LSTM')
        fprintf('  BER (LSTM-DPA-TA) @ max SNR: %.4e\n', lstm_data.BER_scheme_LSTM(end));
    end
    if isfield(lstm_data,'ERR_scheme_LSTM')
        fprintf('  NMSE (LSTM-DPA-TA) @ max SNR: %.4e\n', lstm_data.ERR_scheme_LSTM(end));
    end
end

fprintf('========================================\n');
fprintf('Plotting complete.\n\n');

end