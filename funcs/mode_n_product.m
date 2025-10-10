function Z = mode_n_product(X,Y,n)
    % MODE_N_PRODUCT takes tensor X and compatible matrix Y and performs mode-n product between X and Y.
    % INPUT tensor X, matrix Y.
    % OUTPUT tensor Z.


shape_array = size(X);
Ydims = size(Y); % == J x In
N = numel(shape_array); %order of X



X = mode_n_matricization(X,n);
Z = Y*X;



Z = reshape(Z, [Ydims(1), shape_array(1:n-1), shape_array(n+1:N)]);
Z = permute(Z, [2:n 1 (n+1):N]);

end
%}
