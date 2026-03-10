%% RUN_GRNN_SEEDS_CV.m
clear; clc;
load('prepared.mat','X','y','groups','predictors');

K = 10; 
S = 20; 
baseSeed = 1;

acc_seed  = zeros(S,1);
sens_seed = zeros(S,1);
spec_seed = zeros(S,1);
prec_seed = zeros(S,1);
f1_seed   = zeros(S,1);
auc_seed  = zeros(S,1);
brier_seed = zeros(S,1);

ci_auc_boot_seed = zeros(S,2);
cal_int_seed = zeros(S,1);
cal_slope_seed = zeros(S,1);
Csum_seed = zeros(2,2,S);

spread = 0.2;   % fixed GRNN spread

% Save pooled out-of-fold predictions per seed
y_all_seed    = cell(S,1);
p_all_seed    = cell(S,1);
pred_all_seed = cell(S,1);

for s = 1:S
    
    rng(baseSeed + s, 'twister');
    foldId = stratifiedGroupKFold(y, groups, K, baseSeed + s);

    allC = zeros(2,2,K);

    % Per-seed pooled predictions
    y_all = [];
    p_all = [];

    fold_auc = zeros(K,1);
    fold_brier = zeros(K,1);

    for k = 1:K
        
        te = (foldId == k);
        tr = ~te;

        Xtr = X(tr,:); 
        ytr = y(tr);
        Xte = X(te,:); 
        yte = y(te);

        % Leakage-free imputation
        [Xtr, Xte] = imputePimaWithinFold(Xtr, Xte, predictors, 'median');

        % Leakage-free scaling
        sc = fitMinMaxScaler(Xtr);
        Xtr = applyMinMaxScaler(Xtr, sc);
        Xte = applyMinMaxScaler(Xte, sc);

        % GRNN expects inputs as p x n
        P = Xtr';      % predictors
        T = ytr';      % targets (0/1)

        net = newgrnn(P, T, spread);

        % Predict probabilities
        p_pos = net(Xte');              % 1 x n
        p_pos = max(min(p_pos(:),1),0); % clamp to [0,1]

        % Hard predictions
        pred = double(p_pos >= 0.5);

        % Store pooled out-of-fold predictions
        y_all = [y_all; yte(:)];
        p_all = [p_all; p_pos];

        % Fold-level AUC and Brier
        [~,~,~,AUC] = perfcurve(yte(:), p_pos, 1);
        fold_auc(k) = AUC;
        fold_brier(k) = mean((p_pos - yte(:)).^2);

        % Fold confusion matrix
        allC(:,:,k) = confusionmat(yte, pred, 'Order',[0 1]);
    end

    % Save pooled predictions for this seed
    y_all_seed{s}    = y_all(:);
    p_all_seed{s}    = p_all(:);
    pred_all_seed{s} = double(p_all(:) >= 0.5);

    % Aggregate confusion matrix over folds
    Csum = sum(allC,3);
    Csum_seed(:,:,s) = Csum;

    % Standard metrics from pooled fold predictions
    mm = metricsFromConfMat(Csum);

    acc_seed(s)  = mm.acc;
    sens_seed(s) = mm.sens;
    spec_seed(s) = mm.spec;
    prec_seed(s) = Csum(2,2) / max(1, (Csum(2,2) + Csum(1,2))); % TP / (TP+FP)
    f1_seed(s)   = mm.f1;

    % Probability-based summaries
    auc_seed(s)   = mean(fold_auc);
    brier_seed(s) = mean(fold_brier);

    % Per-seed bootstrap CI
    ci_auc_boot_seed(s,:) = bootstrapAUC(y_all, p_all, 2000, baseSeed + 999*s);

    % Calibration
    [cal_int_seed(s), cal_slope_seed(s)] = calibrationSlopeIntercept(y_all, p_all);

    fprintf('Seed %d | Acc = %.4f\n', s, acc_seed(s));
end

%% Final Report (Mean ± SD across seeds)

fprintf('\n=== GRNN (spread = %.3f) (%d seeds, %d-fold CV) ===\n', spread, S, K);
fprintf('Accuracy = %.4f ± %.4f\n', mean(acc_seed), std(acc_seed));
fprintf('Sensitivity = %.4f ± %.4f\n', mean(sens_seed), std(sens_seed));
fprintf('Specificity = %.4f ± %.4f\n', mean(spec_seed), std(spec_seed));
fprintf('F1-score = %.4f ± %.4f\n', mean(f1_seed), std(f1_seed));
fprintf('Precision = %.4f ± %.4f\n', mean(prec_seed), std(prec_seed));

save('results_grnn_seeds.mat', ...
    'acc_seed','sens_seed','spec_seed','prec_seed','f1_seed','auc_seed', ...
    'brier_seed','ci_auc_boot_seed','cal_int_seed','cal_slope_seed', ...
    'Csum_seed','spread','y_all_seed','p_all_seed','pred_all_seed');

%% Aggregate confusion matrices across seeds
Cgrand = sum(Csum_seed, 3);
TN = Cgrand(1,1); FP = Cgrand(1,2);
FN = Cgrand(2,1); TP = Cgrand(2,2);

[sens_hat, sens_ci] = wilsonCI(TP, TP+FN, 0.05);
[spec_hat, spec_ci] = wilsonCI(TN, TN+FP, 0.05);

fprintf('\n--- Probability-based metrics (across seeds) ---\n');
fprintf('Mean AUC = %.4f ± %.4f\n', mean(auc_seed), std(auc_seed));
fprintf('Mean Brier = %.4f ± %.4f\n', mean(brier_seed), std(brier_seed));

ci_low  = mean(ci_auc_boot_seed(:,1));
ci_high = mean(ci_auc_boot_seed(:,2));
fprintf('AUC 95%% CI (bootstrap; avg across seeds) = [%.4f, %.4f]\n', ci_low, ci_high);

fprintf('Sensitivity (pooled) = %.4f, 95%% CI [%.4f, %.4f]\n', sens_hat, sens_ci(1), sens_ci(2));
fprintf('Specificity (pooled) = %.4f, 95%% CI [%.4f, %.4f]\n', spec_hat, spec_ci(1), spec_ci(2));

fprintf('Calibration intercept = %.4f ± %.4f\n', mean(cal_int_seed), std(cal_int_seed));
fprintf('Calibration slope     = %.4f ± %.4f\n', mean(cal_slope_seed), std(cal_slope_seed));

%% Correct pooled AUC across all seeds
y_pool = [];
p_pool = [];

for s = 1:S
    y_pool = [y_pool; y_all_seed{s}];
    p_pool = [p_pool; p_all_seed{s}];
end

[~,~,~,AUC_pooled] = perfcurve(y_pool, p_pool, 1);
ci_pooled = bootstrapAUC(y_pool, p_pool, 2000, 777);

fprintf('\n--- Pooled subject-level evaluation ---\n');
fprintf('Pooled AUC = %.4f\n', AUC_pooled);
fprintf('Pooled AUC 95%% CI = [%.4f, %.4f]\n', ci_pooled(1), ci_pooled(2));