function m = metricsFromConfMat(C)
% C = [TN FP; FN TP]
TN = C(1,1); FP = C(1,2); FN = C(2,1); TP = C(2,2);

m.acc = (TP+TN) / max(1,(TP+TN+FP+FN));
m.sens = TP / max(1,(TP+FN)); % recall for positive class
m.spec = TN / max(1,(TN+FP));
m.prec = TP / max(1,(TP+FP));
m.f1 = 2*m.prec*m.sens / max(1e-12,(m.prec+m.sens));
end