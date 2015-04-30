function [pred, model] = runLibMF(Ytrain, Ytest, params, options)

    % PROCESS INPUT
    [M,N] = size(Ytrain);

    [Itr, Jtr, Vtr] = prepData(Ytrain, options);
    [Ite, Jte, Vte] = prepData(Ytest, options);

    % RUN LIBMF
    [V, U, bi, bu, avg] = myTrain(Itr,Jtr,Vtr,Ite,Jte,Vte, params.D,params.lp, params.lq, options.eta, options.maxIt, M, N, options.ub, options.ib, options.avg, options.nThreads);

    % PREDICT
    Rhat = V' * U;

    if options.ib
        Rhat = Rhat + repmat(bi', 1,N);
    end
    if options.ub
        Rhat = Rhat + repmat(bu, M,1);
    end
    if options.avg
        Rhat = Rhat + avg;
    end

    % RMSE
    testIdx = sub2ind([M,N], Ite+1, Jte+1);
    pred.yhat = (Rhat(testIdx));
    residual = Vte - pred.yhat;
    pred.yhat = double(pred.yhat);
    pred.rmse = sqrt( dot(residual, residual) / length(residual));

    % OUTPUT
    model.V = double(V);
    model.U = double(U);
    model.bi = double(bi);
    model.bu = double(bu);
    model.avg = avg;
end

function [I, J, V] = prepData(Y, options)
    [I,J,V] = find(Y);
    I = int32(I-1);
    J = int32(J-1);
    if options.transform
        V = single(feval(options.t,V));
    else
        V = single(V);
    end

end
