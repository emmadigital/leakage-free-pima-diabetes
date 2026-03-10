%% RUN_0_prepare_data_from_csv.m
% Tailored for PIMA header:
% Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI,
% DiabetesPedigreeFunction, Age, Outcome

clear; clc;

fileName = 'diabetes.csv';
assert(isfile(fileName)==1, 'Cannot find %s in the current folder.', fileName);

% Auto-detect delimiter (comma vs tab)
firstLine = strtrim(fileread(fileName));
eol = regexp(firstLine, '\r\n|\n|\r', 'match', 'once');
if ~isempty(eol)
    firstLine = extractBefore(firstLine, eol);
end

if contains(firstLine, sprintf('\t'))
    delim = '\t';
else
    delim = ',';
end

opts = detectImportOptions(fileName, 'Delimiter', delim);
T = readtable(fileName, opts);

% Ensure expected columns exist
expected = {'Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin', ...
            'BMI','DiabetesPedigreeFunction','Age','Outcome'};

assert(all(ismember(expected, T.Properties.VariableNames)), ...
    'Column mismatch. Found: %s', strjoin(T.Properties.VariableNames, ', '));

% Predictors in exact order
predictors = expected(1:8);
X = table2array(T(:, predictors));
X = double(X);

% Label
y = double(T.Outcome);
u = unique(y(:));
assert(all(ismember(u,[0 1])), 'Outcome must be binary 0/1. Found: %s', mat2str(u'));

% Remove NaN rows (rare, but safe)
nanRows = any(isnan(X),2) | isnan(y);
if any(nanRows)
    fprintf('Removing %d rows containing NaN.\n', sum(nanRows));
    X(nanRows,:) = [];
    y(nanRows,:) = [];
end

N = size(X,1);
fprintf('Loaded %s | N=%d | p=%d\n', fileName, N, size(X,2));
fprintf('Predictors: %s\n', strjoin(predictors, ', '));

% Duplicate-aware grouping (ensures duplicates stay in same fold)
Z = [X, y];
[~,~,groups] = unique(Z, 'rows', 'stable');

fprintf('Groups created: %d | Max group size: %d\n', numel(unique(groups)), max(histcounts(groups)));

save('prepared.mat','X','y','groups','predictors');

fprintf('Saved prepared.mat (X,y,groups,predictors).\n');