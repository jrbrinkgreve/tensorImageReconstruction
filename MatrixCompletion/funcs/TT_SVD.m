function [tt_cores,rel_error] = TT_SVD(tensor, epsilon)
%TT_SVD(tensor,epsilon) 
%   Algorithm that decomposes a given N-th order tensor into tensor train format
%INPUT:
%   tensor (N-dimensional double):  N-th order tensor
%   epsilon (double):               error-bound for approximation error of TT decomposition
%OUTPUT:
%   tt_cores (cell array with N+1 cells): tensor train (tt) decomposition of a 
%                                   N-th order tensor, where the tt-cores 
%                                   are stored in the first N cells and the 
%                                   location of the norm-core is stored in the 
%                                   N+1 cell. If the tt is not in site_n
%                                   mixed canonical form, the N+1 cell
%                                   contains a 0.
%   error (double):                 actual approximation error of TT
%                                   decomposition

n = ndims(tensor);
tensorshape = size(tensor);


% every cell(1,n) contains one tt-core, which is a r_i x n_i x r_i+1 double
% do not change the datastructure! Otherwise, the testfunction is not
% guaranteed to work.
% The cell(1,n+1) contains the location of the core at the moment. 
% If the norm is stored in the 10th core, the value in the cell(1,d+1) should be 10. 
% If unsure, have a look at the variable tt_dog



tt_cores = cell(1,n+1);
tolerable_error_sq = epsilon^2 * frob_norm(tensor)^2 / n;
ranks = zeros(1,n);
error_sq = 0;
%------ Implement your code below ------

%first step
core_index = 1;
reshaped_tensor = mode_n_matricization(tensor, core_index);
[U, S, V] = svd(reshaped_tensor, 'econ');
s = diag(S);

%rank selection procedure
s_sq = s.^2;
acc_errors = cumsum(s_sq, 'reverse');
included_sv = acc_errors > tolerable_error_sq;
r = sum(included_sv);

%no rank 0
if r == 0
    r = 1;
end

%store
ranks(core_index) = r;

%register error
if r < length(s)
    error_sq = error_sq + sum(s_sq(r+1:end));
end

S = diag(s(1:r));
U = U(:, 1:r);
V = V(:, 1:r);

%save decomposition
tt_cores{n+1} = core_index;
tt_cores{core_index} = reshape(U, [1, tensorshape(core_index), r]);
remainder = S * V';




    

%loop:
for core_index = 2:n-1
    reshaped_tensor = reshape(remainder, [ranks(core_index-1)*tensorshape(core_index), prod(tensorshape(core_index+1:end)) ]);
    [U, S, V] = svd(reshaped_tensor, 'econ');
    s = diag(S);
    
    %rank selection procedure
    s_sq = s.^2;
    acc_errors = cumsum(s_sq, 'reverse');
    included_sv = acc_errors > tolerable_error_sq;
    r = sum(included_sv);
    
    %no rank 0
    if r == 0
        r = 1;
    end
    
    %store
    ranks(core_index) = r;
    
    %register error
    if r < length(s)
        error_sq = error_sq + sum(s_sq(r+1:end));
    end
    
    S = diag(s(1:r));
    U = U(:, 1:r);
    V = V(:, 1:r);

    U_tensorized = reshape(U, [ranks(core_index-1), tensorshape(core_index), ranks(core_index)]);
    tt_cores{n+1} = core_index;
    tt_cores{core_index} = U_tensorized;
    remainder = S * V';
end
    


%last one
tt_cores{core_index+1} = reshape(remainder, [r, tensorshape(core_index+1) , 1] );
tt_cores{n+1} = core_index+1;

%error
rel_error = sqrt(error_sq) / frob_norm(tensor); 
end


