 function [data M m] =scale_new(data,M,m)
[Nb_s Nb_b]=size(data);
if nargin==1
    M=max(data,[],1);
    m=min(data,[],1);
end

data = (data-repmat(m,Nb_s,1))./(repmat(M-m,Nb_s,1));

