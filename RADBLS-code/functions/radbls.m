function [W,M,T,loss] = radbls(A,c,Y,train_num,gamma,alpha,beta,lambda,eta)

[n,m] = size(A);

options = [];
options.NeighborMode = 'KNN';
options.k = 3;
options.WeightMode = 'HeatKernel';

L = construct_L(A',options);

W = pinv(A) * Y;
M = zeros(n,c);
B = 2 * Y - ones(n,c);
T = Y + B .* M;

maxIter = 30;
loss = zeros(maxIter,1);

warning off

D_W = A' * A + beta * A' * L * A;

for iter = 1:maxIter

    W_old = W;
    M_old = M;
    T_old = T;

    U_old = zeros(size(T_old));

    for i = 1:c
        idx = sum(train_num(1:i)) + 1 : sum(train_num(1:i+1));
        mu_i = mean(T_old(idx,:),1);
        U_old(idx,:) = repmat(mu_i,length(idx),1);
    end

    R = Y + B .* M_old;

    G_T = (1 + alpha + lambda) * eye(c) + gamma * (W_old' * W_old);
    D_T = (1 + gamma) * A * W_old + alpha * R + lambda * U_old;

    T = D_T / G_T;

    E_W = gamma * (T' * T);
    C_W = (1 + gamma) * A' * T;

    W = sylvester(D_W,E_W,C_W);

    M = max(B .* (T - Y),0);

    term1 = norm(T - A * W,'fro')^2;
    term2 = alpha * norm(T - (Y + B .* M),'fro')^2;
    term3 = beta * trace(W' * A' * L * A * W);
    term4 = gamma * norm(T * W' - A,'fro')^2;

    omega_c = 0;

    for i = 1:c
        idx = sum(train_num(1:i)) + 1 : sum(train_num(1:i+1));
        T_i = T(idx,:);
        mu_i = mean(T_i,1);
        U_i = repmat(mu_i,length(idx),1);
        omega_c = omega_c + norm(T_i - U_i,'fro')^2;
    end

    term5 = lambda * omega_c;

    loss(iter) = term1 + term2 + term3 + term4 + term5;

    if iter > 1
        if abs(loss(iter) - loss(iter-1)) < eta
            break;
        end
    end

end

loss = loss(1:iter);

end