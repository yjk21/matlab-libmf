% Randomness in LibMF:
% for libMF there are three places where randomization enters:
% 1. initialization: Factors are initialized in [0,1] * 0.1
% 2. permuting input triplets: this may help sgd to converge
% 3. update scheduling: sgd passes over the data in random order
% libMF uses srand48(0) to fix the seed every time. This will only affect 1. and 2.
% When multi-threading is used, 3. may still be non-deterministic, such that iterations differ slightly
rng(1); % control any external randomization (e.g. train-test splits)

% LibMF Naming convention:
% - Q: item factors DxM
% - P: user factors DxN
% where the dimensions are:
M = 4; %users
N = 8; %items
D = 1; %latent dims

lp = 0.0;   % regularization: user factor
lq = 0.0;   % regularization: item factor
eta = 0.02; % learning rate
maxIt = 1200; % sgd max passes over data

% example problem: rank 1 problem  
R =  ones(M,N);
% Completely unobserved users/items will be ignored. The algorithm will just not update these factors, since no ratings exist
R(:,5) = 0;
R(2,:) = 0;
% The input to LibMF are triplets <user> <item> <rating>
% There types have to be int32 int32 float32
% Also, the output will be in single precision
[I, J, V] = find(R);
I = int32(I - 1);
J = int32(J - 1);
Val = single(V);

[V, U, bi, bu, avg] = mexLibMF(I,J,Val,I,J,Val, D,lp, lq, eta, maxIt, M, N, 0, 0,0);

Rhat = V' * U + repmat(bi', 1, N) + repmat(bu, M,1) + avg;
