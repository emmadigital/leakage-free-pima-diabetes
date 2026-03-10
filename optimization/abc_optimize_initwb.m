function bestWB = abc_optimize_initwb(baseNet, Xtr, ytr, Xval, yval, seed)
rng(seed,'twister');

% --- ABC settings (keep modest for speed) ---
nPop = 20;
MaxIt = 10;
limit = 10;

% Weight vector length
w0 = getwb(baseNet);
D = numel(w0);

% Initialize population near base weights
pop = repmat(w0,1,nPop) + 0.1*randn(D,nPop);

fit = zeros(nPop,1);
trial = zeros(nPop,1);

for i=1:nPop
    fit(i) = fitness(pop(:,i));
end

bestWB = pop(:,argmin(fit));

for it=1:MaxIt
    % Employed bees
    for i=1:nPop
        k = randi(nPop); while k==i, k=randi(nPop); end
        phi = randn(D,1)*0.1;
        v = pop(:,i) + phi.*(pop(:,i)-pop(:,k));
        fv = fitness(v);

        if fv < fit(i)
            pop(:,i)=v; fit(i)=fv; trial(i)=0;
        else
            trial(i)=trial(i)+1;
        end
    end

    % Onlooker bees (roulette on inverse fitness)
    P = (1./(fit+1e-12)) / sum(1./(fit+1e-12));
    for t=1:nPop
        i = roulette(P);
        k = randi(nPop); while k==i, k=randi(nPop); end
        phi = randn(D,1)*0.1;
        v = pop(:,i) + phi.*(pop(:,i)-pop(:,k));
        fv = fitness(v);

        if fv < fit(i)
            pop(:,i)=v; fit(i)=fv; trial(i)=0;
        else
            trial(i)=trial(i)+1;
        end
    end

    % Scouts
    for i=1:nPop
        if trial(i) >= limit
            pop(:,i)=w0 + 0.1*randn(D,1);
            fit(i)=fitness(pop(:,i));
            trial(i)=0;
        end
    end

    % Track best
    [fbest, ibest] = min(fit);
    bestWB = pop(:,ibest);

    fprintf('  ABC it %d/%d best val loss=%.5f\n', it, MaxIt, fbest);
end

    function f = fitness(wb)
        net = baseNet;
        net = setwb(net, wb);

        % quick train on training subset only
        Ttr = full(ind2vec(ytr'+1));
        net = configure(net, Xtr', Ttr);
        net.trainParam.epochs = 50;
        net.trainParam.showWindow = false;
        net.divideFcn = 'dividetrain';
        net = train(net, Xtr', Ttr);

        % validation loss (cross-entropy proxy)
        yhat = net(Xval');
        [~,pred] = max(yhat,[],1);
        pred = pred' - 1;

        % simple 0-1 loss; you can change to crossentropy if you want
        f = mean(pred ~= yval);
    end

    function idx = roulette(P)
        r = rand;
        c = cumsum(P);
        idx = find(r <= c,1,'first');
    end

    function i = argmin(v)
        [~,i]=min(v);
    end
end