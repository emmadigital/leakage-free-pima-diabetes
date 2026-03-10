%% STAT_COMPARE_MODELS.m
% Statistical significance testing for diabetes prediction models
% Requires:
%  - acc_seed, auc_seed, sens_seed, spec_seed, f1_seed
%  - y_all_seed, p_all_seed, pred_all_seed
%
% Best practice:
%  - all models should ideally be run with the same number of seeds
%  - the same outer CV partition should be used across models

clear; clc;

models = { ...
    'Logistic Regression', 'results_logreg_seeds.mat'; ...
    'Random Forest',       'results_rf_seeds.mat'; ...
    'Plain FFNN',          'results_plain_seeds.mat'; ...
    'GRNN',                'results_grnn_seeds.mat'; ...
    'BPNN',                'results_bpnn_seeds.mat'; ...
    'ABC-FFNN',            'results_abc_seeds.mat' ...
};


% Pairwise comparisons you want
pairs = {
    'Logistic Regression', 'Random Forest';
    'Logistic Regression', 'Plain FFNN';
    'Logistic Regression', 'GRNN';
    'Logistic Regression', 'BPNN';
    'Logistic Regression', 'ABC-FFNN';
    'Random Forest',       'Plain FFNN';
    'Random Forest',       'GRNN';
    'Random Forest',       'BPNN';
    'Random Forest',       'ABC-FFNN';
    'Plain FFNN',          'GRNN';
    'Plain FFNN',          'BPNN';
    'Plain FFNN',          'ABC-FFNN';
    'GRNN',                'ABC-FFNN'
};

% Load all results
R = struct();
for i = 1:size(models,1)
    modelName = models{i,1};
    modelFile = models{i,2};

    assert(isfile(modelFile), 'Missing file: %s', modelFile);
    R.(matlab.lang.makeValidName(modelName)) = load(modelFile);
end

fprintf('\n============================================\n');
fprintf('STATISTICAL COMPARISON OF MODELS\n');
fprintf('============================================\n');

for p = 1:size(pairs,1)

    modelA = pairs{p,1};
    modelB = pairs{p,2};

    A = R.(matlab.lang.makeValidName(modelA));
    B = R.(matlab.lang.makeValidName(modelB));

    fprintf('\n--------------------------------------------\n');
    fprintf('Comparing: %s  vs  %s\n', modelA, modelB);
    fprintf('--------------------------------------------\n');

    %% A) Multi-seed paired tests
    % Use minimum common number of seeds
    nSeed_auc = min(numel(A.auc_seed), numel(B.auc_seed));
    nSeed_acc = min(numel(A.acc_seed), numel(B.acc_seed));

    fprintf('\n[1] Multi-seed paired comparison\n');

    % AUC
    [~, p_t_auc, ~, stats_t_auc] = ttest(A.auc_seed(1:nSeed_auc), B.auc_seed(1:nSeed_auc));
    p_w_auc = signrank(A.auc_seed(1:nSeed_auc), B.auc_seed(1:nSeed_auc));

    fprintf('AUC  paired t-test: p = %.6f, t(%d) = %.4f\n', ...
        p_t_auc, stats_t_auc.df, stats_t_auc.tstat);
    fprintf('AUC  Wilcoxon signed-rank: p = %.6f\n', p_w_auc);

    % Accuracy
    [~, p_t_acc, ~, stats_t_acc] = ttest(A.acc_seed(1:nSeed_acc), B.acc_seed(1:nSeed_acc));
    p_w_acc = signrank(A.acc_seed(1:nSeed_acc), B.acc_seed(1:nSeed_acc));

    fprintf('ACC  paired t-test: p = %.6f, t(%d) = %.4f\n', ...
        p_t_acc, stats_t_acc.df, stats_t_acc.tstat);
    fprintf('ACC  Wilcoxon signed-rank: p = %.6f\n', p_w_acc);

    %% B) AUC comparison with DeLong test
    fprintf('\n[2] Paired AUC comparison (DeLong test)\n');

    if isfield(A,'y_all_seed') && isfield(A,'p_all_seed') && ...
       isfield(B,'y_all_seed') && isfield(B,'p_all_seed')

        [yPair, pA, pB, predA_pair, predB_pair] = getPairedPredictionsAcrossSeeds(A, B);

        [p_delong, z_delong, aucA, aucB] = delong_roc_test(yPair, pA, pB);

        fprintf('%s AUC = %.4f\n', modelA, aucA);
        fprintf('%s AUC = %.4f\n', modelB, aucB);
        fprintf('DeLong test: z = %.4f, p = %.6f\n', z_delong, p_delong);

    else
        fprintf('Skipped DeLong test: pooled probabilities not saved in one or both result files.\n');
    end

    %% C) McNemar test for paired classification disagreement
    fprintf('\n[3] Paired accuracy comparison (McNemar test)\n');

    if isfield(A,'y_all_seed') && isfield(A,'p_all_seed') && ...
       isfield(B,'y_all_seed') && isfield(B,'p_all_seed')

        [yPair, pA, pB, predA_pair, predB_pair] = getPairedPredictionsAcrossSeeds(A, B);

        statsMc = mcnemar_paired(yPair, predA_pair, predB_pair);


        fprintf('Discordant pairs: n01 = %d, n10 = %d\n', statsMc.n01, statsMc.n10);
        fprintf('McNemar test (%s): statistic = %.4f, p = %.6f\n', ...
            statsMc.method, statsMc.statistic, statsMc.p);

    else
        fprintf('Skipped McNemar test: paired predictions not saved in one or both result files.\n');
    end
end