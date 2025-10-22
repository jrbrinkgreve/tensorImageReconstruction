function X = Ket2Img(T, R, C)

sz = size(T);
dims = numel(sz);

chan = sz(end);
L = dims - 1;  %number of 4-modes 


%spatial size after augmentation
Hf = R / (2^L);
Wf = C / (2^L);

%Build the shape of the permuted tensor that forward produced:
T_new = [repmat(4,1,L), Hf, Wf, chan];

Tperm = reshape(T, T_new);

% [3:(2+L), 1, 2, (2+L+1):end]
sz_before_perm = [Hf, Wf, repmat(4,1,L), chan];
four_idx = 3:(2+L);
ids = [four_idx, 1, 2, length(size(Tperm))];
Tcur = ipermute(Tperm, ids);  % Tcur [Hf, Wf, 4,4,..., C]

for level = L:-1:1
    Hc = size(Tcur,1);
    Wc = size(Tcur,2);
    rest = size(Tcur); res = rest(3:end); rest_prod = prod(res(2:end));

    %  4 mode into 2x2
    dims_expand = [Hc, Wc, res]; 
    % First reshape to view the 4 as (2,2) while preserving linear layout:
    T_expand = reshape(Tcur, [Hc, Wc, 2, 2, rest_prod]);  % [Hc, Wc, 2, 2, rest_prod]

    Tlin = ipermute(T_expand, [2, 4, 1, 3, 5]);   % [2, Hc, 2, Wc, rest_prod]

    newH = Hc * 2;
    newW = Wc * 2;
    if isempty(res)
        Tcur = reshape(Tlin, newH, newW, []);  % final levels will lead to [R,C,chan]
    else
        new_rest = res(2:end); % drop the consumed 4 mode
        Tcur = reshape(Tlin, [newH, newW, new_rest]);
    end
end

X = reshape(Tcur, R, C, chan);
end
