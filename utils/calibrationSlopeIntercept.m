function [cal_int, cal_slope] = calibrationSlopeIntercept(y, p)
% Logistic calibration: logit(p) regressed on outcome
y = double(y(:));
p = max(min(p(:),1-1e-6),1e-6);
lp = log(p./(1-p));                 % logit

% Fit y ~ 1 + lp
mdl = fitglm(lp, y, 'Distribution','binomial','Link','logit');
b = mdl.Coefficients.Estimate;

cal_int = b(1);
cal_slope = b(2);
end