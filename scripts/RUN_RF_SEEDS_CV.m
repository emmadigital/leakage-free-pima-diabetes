%% RUN_RF_SEEDS_CV.m
clear; clc;

load('prepared.mat','X','y','groups','predictors');

K = 10;      % outer folds
S = 20;      % seeds
baseSeed = 1;

auc_seed = zeros(S,1);
acc_seed  = zeros(S,1);
sens_seed = zeros(S,1);
spec_seed = zeros(S,1);
f1_seed   = zeros(S,1);
prec_seed = zeros(S,1);

brier_seed = zeros(S,1);
ci_auc_boot_seed = nan(S,2);
cal_int_seed = zeros(S,1);
cal_slope_seed = zeros(S,1);

Csum_seed = zeros(2,2,S);

y_all_seed    = cell(S,1);
p_all_seed    = cell(S,1);
pred_all_seed = cell(S,1);

% Random Forest settings
nTrees = 200;
minLeaf = 5;

for s = 1:S
    
    rng(baseSeed + s,'twister');
    
    foldId = stratifiedGroupKFold(y, groups, K, baseSeed+s);
    
    allC = zeros(2,2,K);
    
    y_all = [];
    p_all = [];
    
    fold_auc = zeros(K,1);
    fold_brier = zeros(K,1);
    
    for k = 1:K
        
        te = (foldId==k);
        tr = ~te;
        
        Xtr = X(tr,:);
        ytr = y(tr);
        
        Xte = X(te,:);
        yte = y(te);
        
        % ---- Fold-wise imputation ----
        [Xtr, Xte] = imputePimaWithinFold(Xtr, Xte, predictors,'median');
        
        % ---- Fold-wise scaling ----
        % Note: RF does not require scaling, but keep it for consistency
        sc = fitMinMaxScaler(Xtr);
        Xtr = applyMinMaxScaler(Xtr,sc);
        Xte = applyMinMaxScaler(Xte,sc);
        
        % ---- Random Forest ----
        mdl = TreeBagger(nTrees, Xtr, ytr, ...
            'Method','classification', ...
            'MinLeafSize',minLeaf, ...
            'OOBPrediction','off', ...
            'NumPredictorsToSample','all');
        
        % Predicted probabilities
        [~, score] = predict(mdl, Xte);
        
        % TreeBagger returns scores as cell/string labels order
        % Ensure class "1" probability is used
        if iscell(score)
            score = str2double(score);
        end
        
        % If two columns exist, second column corresponds to class 1
        p_pos = score(:,2);
        p_pos = max(min(p_pos,1),0);
        
        pred = double(p_pos >= 0.5);
        
        % ---- Store pooled probabilities ----
        y_all = [y_all; yte(:)];
        p_all = [p_all; p_pos(:)];
        
        % ---- AUC ----
        [~,~,~,AUC] = perfcurve(yte,p_pos,1);
        fold_auc(k) = AUC;
        
        % ---- Brier ----
        fold_brier(k) = mean((p_pos - yte).^2);
        
        % ---- Confusion Matrix ----
        C = confusionmat(yte,pred,'Order',[0 1]);
        allC(:,:,k) = C;
        
    end
    
    % ---- Seed-level aggregation ----
    Csum = sum(allC,3);
    Csum_seed(:,:,s) = Csum;
    
    mm = metricsFromConfMat(Csum);
    
    acc_seed(s)  = mm.acc;
    sens_seed(s) = mm.sens;
    spec_seed(s) = mm.spec;
    f1_seed(s)   = mm.f1;
    
    prec_seed(s) = Csum(2,2) / max(1,(Csum(2,2)+Csum(1,2)));
    
    auc_seed(s) = mean(fold_auc);
    brier_seed(s) = mean(fold_brier);
    
    % ---- Bootstrap AUC CI ----
    ci_auc_boot_seed(s,:) = bootstrapAUC(y_all,p_all,2000,baseSeed+999*s);
    
    % ---- Calibration ----
    [cal_int_seed(s),cal_slope_seed(s)] = calibrationSlopeIntercept(y_all,p_all);
    
    % ---- Save pooled predictions ----
    y_all_seed{s} = y_all;
    p_all_seed{s} = p_all;
    pred_all_seed{s} = double(p_all >= 0.5);
    
    fprintf('Seed %d | Acc=%.4f\n',s,acc_seed(s));
    
end

%% Final Results

fprintf('\n=== Random Forest (%d seeds, %d-fold CV) ===\n',S,K);

fprintf('Accuracy = %.4f ± %.4f\n',mean(acc_seed),std(acc_seed));
fprintf('Sensitivity = %.4f ± %.4f\n',mean(sens_seed),std(sens_seed));
fprintf('Specificity = %.4f ± %.4f\n',mean(spec_seed),std(spec_seed));
fprintf('F1-score = %.4f ± %.4f\n',mean(f1_seed),std(f1_seed));
fprintf('Precision = %.4f ± %.4f\n',mean(prec_seed),std(prec_seed));

fprintf('\n--- Probability metrics ---\n');
fprintf('Mean AUC = %.4f ± %.4f\n',mean(auc_seed),std(auc_seed));
fprintf('Mean Brier = %.4f ± %.4f\n',mean(brier_seed),std(brier_seed));

fprintf('Calibration intercept = %.4f ± %.4f\n',mean(cal_int_seed),std(cal_int_seed));
fprintf('Calibration slope     = %.4f ± %.4f\n',mean(cal_slope_seed),std(cal_slope_seed));

save('results_rf_seeds.mat', ...
    'acc_seed','sens_seed','spec_seed','prec_seed','f1_seed','auc_seed', ...
    'ci_auc_boot_seed','Csum_seed','y_all_seed','p_all_seed','pred_all_seed', ...
    'cal_int_seed','cal_slope_seed','brier_seed','nTrees','minLeaf');