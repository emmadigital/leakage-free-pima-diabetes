function [Xtr_imp, Xte_imp, imp] = imputePimaWithinFold(Xtr, Xte, predictorNames, method)
% imputePimaWithinFold
% Leakage-free imputation for Pima Indians Diabetes dataset.
% 1) Treats physiologically invalid zeros as missing (NaN) for selected vars.
% 2) Learns imputation statistics ONLY on Xtr.
% 3) Applies same statistics to Xtr and Xte.
%
% Inputs
%   Xtr, Xte          : numeric matrices (n x p)
%   predictorNames    : 1xp cellstr, e.g. {'Pregnancies','Glucose',...}
%                       If empty [], uses default Pima column order:
%                       [Pregnancies Glucose BloodPressure SkinThickness Insulin BMI DPF Age]
%   method            : 'median' (recommended) or 'mean'
%
% Outputs
%   Xtr_imp, Xte_imp  : imputed matrices
%   imp               : struct with fields .method .cols .values and .names

if nargin < 3 || isempty(predictorNames)
    predictorNames = {'Pregnancies','Glucose','BloodPressure','SkinThickness', ...
                      'Insulin','BMI','DiabetesPedigreeFunction','Age'};
end
if nargin < 4 || isempty(method)
    method = 'median';
end

Xtr_imp = Xtr;
Xte_imp = Xte;

% Columns where 0 is physiologically implausible in Pima and usually treated as missing
invalidZeroVars = {'Glucose','BloodPressure','SkinThickness','Insulin','BMI'};
cols = find(ismember(predictorNames, invalidZeroVars));

% Convert 0 -> NaN in BOTH train and test (rule is fixed a priori; no leakage)
for c = cols
    Xtr_imp(Xtr_imp(:,c)==0, c) = NaN;
    Xte_imp(Xte_imp(:,c)==0, c) = NaN;
end

% Learn imputation value on TRAIN only
impVals = zeros(1, numel(cols));
for j = 1:numel(cols)
    c = cols(j);
    xc = Xtr_imp(:,c);

    if strcmpi(method,'median')
        v = median(xc, 'omitnan');
    elseif strcmpi(method,'mean')
        v = mean(xc, 'omitnan');
    else
        error('Unknown method. Use ''median'' or ''mean''.');
    end

    % Fallback if a column is entirely NaN in training fold (rare)
    if isnan(v)
        v = 0; % conservative fallback; you may replace with global prior if desired
    end
    impVals(j) = v;

    % Apply to train and test
    Xtr_imp(isnan(Xtr_imp(:,c)), c) = v;
    Xte_imp(isnan(Xte_imp(:,c)), c) = v;
end

imp.method = lower(method);
imp.cols   = cols;
imp.values = impVals;
imp.names  = predictorNames(cols);

end