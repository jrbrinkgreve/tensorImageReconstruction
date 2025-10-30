function X = mode_n_matricization(X,n)
% MODE_K_MATRICIZATION takes a tensor X as input and a mode n such
% that n<ndims(X) and returns the mode-n matricization of X.
% INPUTS tensor X, mode n.
% OUPUT mode-n matricization of X.


shape_array = size(X);
N = numel(shape_array);
X = permute(X, [n 1:(n-1), (n+1:N)] );
X = reshape(X, [shape_array(n), prod(shape_array) / shape_array(n)]);

end