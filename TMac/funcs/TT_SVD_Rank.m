function [cores,RankX] = TT_SVD_Rank(X,th)
N =  ndims(X);
cores = cell(1,N);
RankX = zeros(1,N-1);
sizX = size(X);

T = reshape(X,[sizX(1) prod(sizX(2:end))]);
for k = 1:N-1    
    [cores{k}, S, V, tau] = svd_RankEstimate(T,th);    

    if k==1
        cores{k} = reshape(cores{k},[sizX(k) tau]);
    else
        cores{k} = reshape(cores{k},[size(T,1)/sizX(k) sizX(k) tau]);
    end
    T = S*V;    
    T = reshape(T,[tau sizX(k+1:end)]);
    if k<N-1
        T = reshape(T,[tau*sizX(k+1) prod(sizX((k+2):end))]);            
    end
    RankX(k) = tau;    
end
cores{N} = T;

end

function [U, S, V, tau]=svd_RankEstimate(T,th)
[U, S, V]=svd(T, 'econ');
V=V';
tau = 0;
szS = size(S,1); 
%smax = max(diag(S));
for k = 1:szS
    if S(k,k)/S(1,1)>th
        tau = tau+1;
    end
end
tau = max(tau,2); 
U = U(:,1:tau);
S = S(1:tau,1:tau);
V = V(1:tau,:);
fprintf('The original rank is %u and truncated rank is %u\n',szS,tau)
end