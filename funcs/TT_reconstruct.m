function tensor = TT_reconstruct(tt_cores)
%TT_reconstruct(tt,n) 
%   Algorithm that reconstructs a N-th order tensor from the tensor train
%   decomposition
%      
%INPUT:
%   tt (cell array with N+1 cells):  tensor train (tt) decomposition of a 
%                                   N-th order tensor, where the tt-cores 
%                                   are stored in the first N cells and the 
%                                   location of the norm-core is stored in the 
%                                   N+1 cell. If the tt is not in site_n
%                                   mixed canonical form, the N+1 cell
%                                   contains a 0.
%OUTPUT:
%   tensor (N-dimensional double):  N-th order tensor reconstructed from tt. 
%------ Implement your code below ------



%
[~, n] = size(tt_cores);


%last entry in the cell array tt_cores is the location of the norm
%get tt sizes
sz = zeros(n-1, 1);
for i = 1:n-1
[ ~, sz(i), ~] = size(tt_cores{i});
end



tensor = tt_cores{n-1};
% Contract from right to left
for core_index = n-2:-1:1
    B = mode_n_matricization(tensor, 1);
    B = B.';
    A = tt_cores{core_index};
    tensor = mode_n_product(A, B, 3);
end

% Reshape to final size
tensor = reshape(tensor, sz.');

end



