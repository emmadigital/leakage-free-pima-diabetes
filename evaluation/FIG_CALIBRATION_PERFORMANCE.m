%% FIG_CALIBRATION_PERFORMANCE.m
clear; clc; close all;

models = { ...
    'LOGREG', 'results_logreg_seeds.mat'; ...
    'RF',       'results_rf_seeds.mat'; ...
    'Plain FFNN',          'results_plain_seeds.mat'; ...
    'GRNN',                'results_grnn_seeds.mat'; ...
    'BPNN',                'results_bpnn_seeds.mat'; ...
    'ABC-FFNN',            'results_abc_seeds.mat' ...
};

modelNames = {};
cal_slope_mu = [];
cal_slope_sd = [];
cal_int_mu   = [];
cal_int_sd   = [];
brier_mu     = [];
brier_sd     = [];

for i = 1:size(models,1)
    file = models{i,2};

    if ~isfile(file)
        warning('Missing file: %s. Skipping.', file);
        continue;
    end

    S = load(file);

    req = {'cal_slope_seed','cal_int_seed','brier_seed'};
    if ~all(isfield(S, req))
        warning('File %s is missing one or more calibration fields. Skipping.', file);
        continue;
    end

    modelNames{end+1,1} = models{i,1}; %#ok<AGROW>
    cal_slope_mu(end+1,1) = mean(S.cal_slope_seed); %#ok<AGROW>
    cal_slope_sd(end+1,1) = std(S.cal_slope_seed); %#ok<AGROW>

    cal_int_mu(end+1,1) = mean(S.cal_int_seed); %#ok<AGROW>
    cal_int_sd(end+1,1) = std(S.cal_int_seed); %#ok<AGROW>

    brier_mu(end+1,1) = mean(S.brier_seed); %#ok<AGROW>
    brier_sd(end+1,1) = std(S.brier_seed); %#ok<AGROW>
end

nM = numel(modelNames);
assert(nM > 0, 'No valid result files containing calibration metrics were found.');

fig = figure('Color','w','Position',[100 100 1400 420]);

% Panel 1: slope
subplot(1,3,1);
bar(cal_slope_mu, 'FaceColor',[0.20 0.45 0.85], 'EdgeColor','none'); hold on;
errorbar(1:nM, cal_slope_mu, cal_slope_sd, 'k', ...
    'LineStyle','none', 'LineWidth',1.4, 'CapSize',8);
yline(1.0, '--', 'Ideal', 'Color',[0.5 0.5 0.5], 'LineWidth',1.2);
set(gca,'XTick',1:nM,'XTickLabel',modelNames,'FontSize',12,'LineWidth',1.1,'TickLabelInterpreter','none');
xtickangle(20);
ylabel('Calibration slope');
title('Calibration Slope','FontWeight','bold');
grid on; box off;

% Panel 2: intercept
subplot(1,3,2);
bar(cal_int_mu, 'FaceColor',[0.85 0.33 0.10], 'EdgeColor','none'); hold on;
errorbar(1:nM, cal_int_mu, cal_int_sd, 'k', ...
    'LineStyle','none', 'LineWidth',1.4, 'CapSize',8);
yline(0.0, '--', 'Ideal', 'Color',[0.5 0.5 0.5], 'LineWidth',1.2);
set(gca,'XTick',1:nM,'XTickLabel',modelNames,'FontSize',12,'LineWidth',1.1,'TickLabelInterpreter','none');
xtickangle(20);
ylabel('Calibration intercept');
title('Calibration Intercept','FontWeight','bold');
grid on; box off;

% Panel 3: Brier
subplot(1,3,3);
bar(brier_mu, 'FaceColor',[0.47 0.67 0.19], 'EdgeColor','none'); hold on;
errorbar(1:nM, brier_mu, brier_sd, 'k', ...
    'LineStyle','none', 'LineWidth',1.4, 'CapSize',8);
set(gca,'XTick',1:nM,'XTickLabel',modelNames,'FontSize',12,'LineWidth',1.1,'TickLabelInterpreter','none');
xtickangle(20);
ylabel('Brier score');
title('Brier Score','FontWeight','bold');
grid on; box off;

sgtitle('Calibration Performance under Leakage-Free Cross-Validation', ...
    'FontWeight','bold','FontSize',15);

exportgraphics(fig, 'FIG_Calibration_Performance.png', 'Resolution', 600);
exportgraphics(fig, 'FIG_Calibration_Performance.pdf', 'ContentType', 'vector');

disp('Saved: FIG_Calibration_Performance.png and FIG_Calibration_Performance.pdf');