rng(1)
M = 4;
N = 8;
D = 1;

lp = 0.0;
lq = 0.0;
eta = 0.02;
maxIt = 1200;
R =  ones(M,N);
[I, J, V] = find(R);
I = int32(I - 1);
J = int32(J - 1);
Val = single(V);

[V, U, bi, bu, avg] = mexLibMF(I,J,Val,I,J,Val, D,lp, lq, eta, maxIt, M, N);

Rhat = V' * U + repmat(bi', 1, N) + repmat(bu, M,1) + avg;
