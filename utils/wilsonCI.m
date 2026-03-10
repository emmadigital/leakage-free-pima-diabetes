function [phat, ci] = wilsonCI(x, n, alpha)
% Wilson score interval for binomial proportion
if nargin < 3, alpha = 0.05; end
phat = x / max(1,n);

z = norminv(1 - alpha/2);
den = 1 + z^2/n;
center = (phat + z^2/(2*n)) / den;
half = (z/den) * sqrt((phat*(1-phat) + z^2/(4*n))/n);

ci = [max(0, center-half), min(1, center+half)];
end