function sz = TT_get_size(tt)
%TT_get_size (tt,n) 
%   Algorithm that returns the size of a tt
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
%   sz (N x 3 double):              size of TT decomposition, see example
%                                   figure 
%------ Implement your code below ------

[~,n] = size(tt);
sz = zeros(n-1, 1);

for i = 1:n-1

[ ~, sz(i), ~] = size(tt{i});


end