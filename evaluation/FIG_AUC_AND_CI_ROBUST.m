%% FIG_AUC_AND_CI_ROBUST.m
% Publication-ready forest plot of AUC with 95% CI
% Uses mean AUC across seeds and averaged bootstrap CI bounds

clear; clc; close all;

models = { ...
   'LOGREG', 'results_logreg_seeds.mat'; ...
   'Random Forest','results_rf_seeds.mat'; ...
   'Plain FFNN', 'results_plain_seeds.mat'; ...
   'GRNN',       'results_grnn_seeds.mat'; ...
   'BPNN',       'results_bpnn_seeds.mat'; ...
   'ABC-FFNN',   'results_abc_seeds.mat' ...
};

nM = size(models,1);

auc_mean = nan(nM,1);
auc_sd   = nan(nM,1);
ci_low   = nan(nM,1);
ci_high  = nan(nM,1);

for i = 1:nM
    file = models{i,2};
    assert(isfile(file), 'Missing file: %s', file);
    S = load(file);

    assert(isfield(S,'auc_seed'), 'File %s must contain auc_seed', file);
    auc_mean(i) = mean(S.auc_seed);
    auc_sd(i)   = std(S.auc_seed);

    assert(isfield(S,'ci_auc_boot_seed'), ...
        'File %s must contain ci_auc_boot_seed', file);

    ci_low(i)  = mean(S.ci_auc_boot_seed(:,1));
    ci_high(i) = mean(S.ci_auc_boot_seed(:,2));
end

% Sort by AUC descending
[auc_mean, idx] = sort(auc_mean, 'descend');
auc_sd   = auc_sd(idx);
ci_low   = ci_low(idx);
ci_high  = ci_high(idx);
modelNames = models(idx,1);

%% ---- Forest Plot ----
fig = figure('Color','w','Position',[100 100 1000 520]);
ax = axes(fig);
hold(ax,'on');

y = 1:nM;

% Plot CIs
for i = 1:nM
    plot([ci_low(i), ci_high(i)], [y(i), y(i)], ...
        'k-', 'LineWidth', 2);
end

% Plot mean AUC points
scatter(auc_mean, y, 80, 'filled', ...
    'MarkerFaceColor', [0.85 0.33 0.10], ...
    'MarkerEdgeColor', 'k');

% Add labels to the right
for i = 1:nM
    txt = sprintf('AUC = %.3f  [%.3f, %.3f]', auc_mean(i), ci_low(i), ci_high(i));
    text(ci_high(i) + 0.003, y(i), txt, ...
        'VerticalAlignment','middle', ...
        'FontSize',11, ...
        'Interpreter','none');
end

% Formatting
set(ax, 'YTick', y, ...
        'YTickLabel', modelNames, ...
        'YDir', 'reverse', ...
        'FontSize', 13, ...
        'LineWidth', 1.2, ...
        'TickLabelInterpreter','none');

xlabel('Area Under the ROC Curve (AUC)', 'FontSize', 14);
title('Discrimination Performance with 95% Confidence Intervals', ...
      'FontWeight','bold', 'FontSize', 15);

grid(ax, 'on');
box(ax, 'off');

% Better axis limits
xmin = min(ci_low) - 0.02;
xmax = max(ci_high) + 0.10;
xlim([xmin xmax]);
ylim([0.5 nM+0.5]);

% Add vertical reference line at 0.80 (optional benchmark)
xline(0.80, '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.2);

% Export
exportgraphics(fig, 'FIG_AUC_95CI_Forest_Upgraded.png', 'Resolution', 600);
exportgraphics(fig, 'FIG_AUC_95CI_Forest_Upgraded.pdf', 'ContentType', 'vector');

disp('Saved: FIG_AUC_95CI_Forest_Upgraded.png and FIG_AUC_95CI_Forest_Upgraded.pdf');