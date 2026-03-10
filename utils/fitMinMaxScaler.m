function scaler = fitMinMaxScaler(Xtrain)
% returns struct with min/max per feature
scaler.xmin = min(Xtrain,[],1);
scaler.xmax = max(Xtrain,[],1);
% avoid divide by zero
scaler.range = scaler.xmax - scaler.xmin;
scaler.range(scaler.range==0) = 1;
end