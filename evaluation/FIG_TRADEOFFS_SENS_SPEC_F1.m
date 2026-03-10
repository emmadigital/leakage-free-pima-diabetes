%% FIG_TRADEOFFS_SENS_SPEC_F1.m
% Publication-ready grouped bar chart:
% Diagnostic Trade-offs: Sensitivity, Specificity, and F1-Score

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

sens = nan(nM,1);
spec = nan(nM,1);
f1   = nan(nM,1);

for i = 1:nM
    file = models{i,2};
    assert(isfile(file), 'Missing file: %s', file);

    S = load(file);
    req = {'sens_seed','spec_seed','f1_seed'};
    for r = 1:numel(req)
        assert(isfield(S, req{r}), 'File %s must contain %s', file, req{r});
    end

    sens(i) = mean(S.sens_seed);
    spec(i) = mean(S.spec_seed);
    f1(i)   = mean(S.f1_seed);
end

% Sort by F1 descending (recommended for readability)
[~, idx] = sort(f1, 'descend');
sens = sens(idx);
spec = spec(idx);
f1   = f1(idx);
modelNames = models(idx,1);

M = [sens, spec, f1];

%% ---- Plot ----
fig = figure('Color','w','Position',[100 100 1150 540]);
ax = axes(fig);
hold(ax,'on');

b = bar(ax, M, 'grouped', 'BarWidth', 0.78);

b(1).FaceColor = [0.85 0.33 0.10]; % Sensitivity
b(2).FaceColor = [0.00 0.45 0.74]; % Specificity
b(3).FaceColor = [0.47 0.67 0.19]; % F1-score

set(ax, ...
    'XTick', 1:nM, ...
    'XTickLabel', modelNames, ...
    'FontSize', 13, ...
    'LineWidth', 1.2, ...
    'TickLabelInterpreter', 'none');

xtickangle(ax, 20);
ylabel(ax, 'Metric value', 'FontSize', 14);
title(ax, 'Diagnostic Trade-Offs: Sensitivity, Specificity, and F1-Score', ...
    'FontWeight','bold', 'FontSize', 15);

ylim(ax, [0.40 1.00]);
xlim(ax, [0.5 nM+0.5]);
grid(ax, 'on');
box(ax, 'off');

leg = legend(ax, {'Sensitivity','Specificity','F1-score'}, ...
    'Location','northeast', 'Box','off');
leg.FontSize = 12;

% Numeric labels
labelOffset = 0.012;
for j = 1:size(M,2)
    for i = 1:size(M,1)
        xj = b(j).XEndPoints(i);
        yj = M(i,j);
        text(ax, xj, yj + labelOffset, sprintf('%.3f', yj), ...
            'HorizontalAlignment','center', ...
            'VerticalAlignment','bottom', ...
            'FontSize', 10.5, ...
            'FontWeight','bold');
    end
end

exportgraphics(fig, 'FIG_Tradeoffs_Sens_Spec_F1_Upgraded.png', 'Resolution', 600);
exportgraphics(fig, 'FIG_Tradeoffs_Sens_Spec_F1_Upgraded.pdf', 'ContentType', 'vector');

disp('Saved: FIG_Tradeoffs_Sens_Spec_F1_Upgraded.png and FIG_Tradeoffs_Sens_Spec_F1_Upgraded.pdf');