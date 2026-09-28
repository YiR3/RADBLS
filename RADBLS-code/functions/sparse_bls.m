function wk = sparse_bls(Z,X,lambda,itrs)
% ¦Ñ=1
N1 = size(Z,2);
d = size(X,2);
x = zeros(N1,d);
wk = x; 
ok=x;
uk=x;
L1=eye(N1)/((Z') * Z+eye(N1));
L2=L1*Z'*X;

for i = 1:itrs
    tempc=ok-uk;
    ck =  L2+L1*tempc;
    ok=shrinkage(ck+uk, lambda);
    uk=uk+(ck-ok);
    wk=ok;
end
end
function z = shrinkage(a, kappa)
    z = max( a - kappa,0 ) - max( -a - kappa ,0);
end
% function p = objective(A, b, lam, x, z)
%     p = ( 1/2*sum((A*x - b).^2) + lambda*norm(z,1) );
% end
% % toc

