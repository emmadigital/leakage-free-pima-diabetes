function Xs = applyMinMaxScaler(X, scaler)
Xs = (X - scaler.xmin) ./ scaler.range;
end