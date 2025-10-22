
function T = Img2Ket(X, Nway)


[H, W, C] = size(X);


num4 = numel(Nway) - 1;     
Tcur = X;
disp(num4)
for level = 1:num4
    Hc = size(Tcur,1);
    Wc = size(Tcur,2);
    rest = size(Tcur); rest = rest(3:end);        % sizes of remaining trailing dims
    rest_prod = prod(rest);

    % separate each 2x2 block
    dims1 = [2, Hc/2, 2, Wc/2, rest_prod];
    Tlin = reshape(Tcur, dims1);                    % [2, Hc/2, 2, Wc/2, rest_prod]

    Tper = permute(Tlin, [2, 4, 1, 3, 5]);          % [Hc/2, Wc/2, 2, 2, rest_prod]

    % collapse 2x2 into single dimension 4
    dims2 = [Hc/2, Wc/2, 4, rest];
    Tnext = reshape(Tper, dims2);                   % [Hc/2, Wc/2, 4, rest]
    Tcur = Tnext;
end

% Now Tcur is [Hf, Wf, 4, 4, ..., C]
sz = size(Tcur);
four_idx = 3:(2+num4);
perm = [four_idx, 1, 2, length(sz)];   % bring all 4-modes to the front
Tperm = permute(Tcur, perm);

T = reshape(Tperm, Nway);
end
