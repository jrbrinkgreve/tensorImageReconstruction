clearvars; close all; clc;

img = double(imread("4.2.07.tiff"));
dims = [repmat(4,1,9), 3];      
N = numel(dims);

sample = 0.6;                   
maxiter = 300;                 
tolerance = 1e-6;               
reg_eps = 1e-6;                

cutoffs = logspace(log10(1e-3), log10(1e-1), 20);  

RSEs = nan(size(cutoffs));
times = nan(size(cutoffs));

X_orig = reshape(img, dims);
numelX = prod(dims);

obs = randsample(numelX, round(sample * numelX));
obs = sort(obs);                 
ind = obs;                      
Xn = X_orig(ind);

mask = zeros(dims);
mask(ind) = 1;
img_obs = mask .* X_orig;


lambda = zeros(1,N-1);
IL = dims(1);
for k = 1:N-1
    IR = prod(dims(k+1:end));
    lambda(k) = min(IL,IR);
    IL = IL * dims(k+1);
end
lambda = lambda / sum(lambda);


norm_full = norm(X_orig(:));

function [M_final, RSE, elapsed] = run_tmactt_one(X_orig,img_obs, dims, N, ind, Xn, lambda, cutoff, ...
                                                  maxiter, tolerance, reg_eps, norm_full)
    all_idx = (1:prod(dims))';
    values_interp = interp1(ind(:), double(Xn(:)), all_idx, 'linear', 'extrap');
    values_interp = max(min(values_interp, 255), 0);  
    M = reshape(values_interp, dims);
    M(ind) = Xn;  
    [~, rank_vec] = TT_SVD_Rank(M, cutoff);
 
    

    if isscalar(rank_vec)
        rank_vec = repmat(rank_vec, 1, N-1);
    elseif numel(rank_vec) < (N-1)
        rank_vec = repmat(rank_vec(1), 1, N-1);
    else
        rank_vec = rank_vec(1:N-1);
    end



    Xc = cell(1, N-1);
    Yc = cell(1, N-1);
    dimL = zeros(1, N-1);
    dimR = zeros(1, N-1);
    IL = 1;
    for k = 1:N-1
        dimL(k) = IL * dims(k);
        dimR(k) = prod(dims) / dimL(k);
        r = max(1, round(rank_vec(k)));  
        Xc{k} = randn(dimL(k), r);
        Yc{k} = randn(r, dimR(k));
        IL = dimL(k);
    end

    tic;
    relerr = inf;
    prevM = M;
    kiter = 0;
    while (relerr > tolerance) && (kiter < maxiter)
        kiter = kiter + 1;
        Mcur = M;

        for n = 1:N-1
            Mn = reshape(M, [size(Xc{n},1), size(Yc{n},2)]);   % balanced unfolding view
            % update X
            Xc{n} = Mn * Yc{n}';
            Xsq = Xc{n}' * Xc{n};
            rhs = Xc{n}' * Mn;
            Yc{n} = (Xsq + reg_eps * eye(size(Xsq))) \ rhs;
        end

        M_agg = lambda(1) * reshape(Xc{1} * Yc{1}, dims);
        for n = 2:N-1
            Mn = reshape(Xc{n} * Yc{n}, dims);
            M_agg = M_agg + lambda(n) * Mn;
        end
        M = M_agg;
        M(ind) = Xn;   

        relerr = norm(M(:) - Mcur(:)) / (norm_full + eps);

        if kiter > 2 && relerr > 10
            warning('Diverging at iteration %d for cutoff %.3g. Breaking.', kiter, cutoff);
            break;
        end
    end
    elapsed = toc;

    M_final = M;
    RSE = norm(M(:) - X_orig(:)) / (norm_full + eps);
end

fprintf('Running TMac-TT sweep over %d cutoffs...\n', numel(cutoffs));
for i = 1:numel(cutoffs)
    cutoff = cutoffs(i);
    fprintf('Cutoff %2d / %2d: %.4g ... ', i, numel(cutoffs), cutoff);
    try
        [Mrec, RSEs(i), times(i)] = run_tmactt_one(X_orig,img_obs, dims, N, ind, Xn, lambda, ...
                                                   cutoff, maxiter, tolerance, reg_eps, norm_full);
        fprintf('RSE=%.6f, time=%.2fs\n', RSEs(i), times(i));
    catch ME
        fprintf('FAILED: %s\n', ME.message);
        RSEs(i) = NaN;
        times(i) = NaN;
    end
end

[bestRSE, idxBest] = min(RSEs);
bestCutoff = cutoffs(idxBest);

fprintf('\nBest cutoff = %.6g (index %d) with RSE = %.6f\n', bestCutoff, idxBest, bestRSE);

figure();
semilogx(cutoffs, RSEs, '-o', 'LineWidth', 1.4);
grid on;
xlabel('Cutoff');
ylabel('RSE (relative reconstruction error)');
title('Cutoff sweep: RSE vs cutoff');
hold on;
semilogx(bestCutoff, bestRSE, 'r*', 'MarkerSize', 12);
legend('RSE', sprintf('Best cutoff = %.3g', bestCutoff), 'Location', 'best');

fprintf('Reconstructing with best cutoff %.6g for visualization...\n', bestCutoff);
[Mrec_best, RSE_best_vis, tbest] = run_tmactt_one(X_orig, img_obs,dims, N, ind, Xn, lambda, ...
                                                  bestCutoff, maxiter, tolerance, reg_eps, norm_full);
Img_recovered = reshape(Mrec_best, size(img));
Img_original = reshape(X_orig, size(img));

figure();
imshow(uint8(Img_recovered));
title(sprintf('Reconstructed (best cutoff %.3g) — RSE = %.6f', bestCutoff, RSE_best_vis));

figure();
imshow(uint8(Img_original));
title('Original image');

