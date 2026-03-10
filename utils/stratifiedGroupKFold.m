function foldId = stratifiedGroupKFold(y, groups, K, seed)
% Returns foldId per sample (1..K), ensuring groups do not split across folds
% Stratifies by y at group level using majority label in each group.

if nargin < 4, seed = 1; end
rng(seed,'twister');

ug = unique(groups);
G = numel(ug);

% Determine group label (majority)
gy = zeros(G,1);
for i = 1:G
    idx = (groups == ug(i));
    gy(i) = mode(y(idx));
end

foldOfGroup = zeros(G,1);

for cls = [0 1]
    gcls = find(gy==cls);
    gcls = gcls(randperm(numel(gcls)));
    % round-robin assign groups to folds
    for j = 1:numel(gcls)
        foldOfGroup(gcls(j)) = mod(j-1,K)+1;
    end
end

% Map back to samples
foldId = zeros(size(y));
for i = 1:G
    foldId(groups==ug(i)) = foldOfGroup(i);
end
end