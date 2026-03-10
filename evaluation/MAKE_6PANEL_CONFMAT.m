%% MAKE_6PANEL_CONFMAT.m
% Publication-ready 2x3 confusion matrix figure
% Row-normalized confusion matrices pooled across seeds

clear; clc; close all;

models = { ...
    'LOGREG',        'results_logreg_seeds.mat'; ...
    'Random Forest', 'results_rf_seeds.mat'; ...
    'Plain FFNN',    'results_plain_seeds.mat'; ...
    'BPNN',          'results_bpnn_seeds.mat'; ...
    'GRNN',          'results_grnn_seeds.mat'; ...
    'ABC-FFNN',      'results_abc_seeds.mat' ...
};

nM = size(models,1);

fig = figure('Color','w','Position',[100 100 1800 950]);
tiledlayout(2,3,'Padding','compact','TileSpacing','compact');

for i = 1:nM
    
    modelName = models{i,1};
    file = models{i,2};

    assert(isfile(file), 'Missing file: %s', file);
    S = load(file);

    assert(isfield(S,'Csum_seed'), ...
        'File %s must contain Csum_seed', file);

    % Sum confusion matrices across seeds
    Cgrand = sum(S.Csum_seed, 3);

    % Row-normalization
    Crow = Cgrand ./ max(sum(Cgrand,2),1) * 100;

    nexttile;
    imagesc(Crow);
    axis square;
    colormap(parula);
    caxis([0 100]);
    colorbar;

    title(modelName, 'FontWeight','bold', 'FontSize',16);

    xticks([1 2]);
    yticks([1 2]);
    xticklabels({'Pred 0','Pred 1'});
    yticklabels({'True 0','True 1'});
    set(gca, 'FontSize',12, 'LineWidth',1.0);

    % Text annotation inside cells
    for r = 1:2
        for c = 1:2
            txt = sprintf('%.1f%%\n(n=%d)', Crow(r,c), Cgrand(r,c));

            % Choose text color based on background intensity
            if Crow(r,c) > 55
                txtColor = 'k';
            else
                txtColor = 'w';
            end

            text(c, r, txt, ...
                'HorizontalAlignment','center', ...
                'VerticalAlignment','middle', ...
                'FontSize',12, ...
                'FontWeight','bold', ...
                'Color',txtColor);
        end
    end
end

sgtitle('Row-Normalized Confusion Matrices (Pooled Across Seeds)', ...
    'FontWeight','bold', 'FontSize',18);

exportgraphics(fig,'FIG_6Panel_Confusion_Matrices.png','Resolution',600);
exportgraphics(fig,'FIG_6Panel_Confusion_Matrices.pdf','ContentType','vector');

disp('Saved: FIG_6Panel_Confusion_Matrices.png and FIG_6Panel_Confusion_Matrices.pdf');