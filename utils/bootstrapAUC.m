function ci = bootstrapAUC(y, p, B, seed)
% Bootstrap 95% CI for AUC
% y: true labels (0/1)
% p: predicted probabilities for class 1
% B: number of bootstrap samples (e.g., 2000)
% seed: random seed

if nargin < 3
    B = 2000;
end
if nargin < 4
    seed = 1;
end

rng(seed,'twister');

n = numel(y);
aucs = zeros(B,1);

for b = 1:B
    idx = randi(n, n, 1);
    yb = y(idx);
    pb = p(idx);

    % perfcurve fails if only one class present
    if numel(unique(yb)) < 2
        aucs(b) = NaN;
        continue;
    end

    [~,~,~,aucs(b)] = perfcurve(yb, pb, 1);
end

aucs = aucs(~isnan(aucs));
ci = prctile(aucs, [2.5 97.5]);
end