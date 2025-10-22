clc
clear
close all
addpath('./funcs');

%% Load data and parameters
orig_img = double(imread("4.2.07.tiff"));
[R, C, ~] = size(orig_img);

dims = [4 4 4 4 4 4 4 4 4 3];    
N = numel(dims);

SR = 0.1;   %sampling              
sampleFraction = SR;

tol = 1e-4;                     % Convergence tolerance
maxit = 1000;                    % Maximum iterations

%T_original = Img2Ket(orig_img, dims);
T_original = reshape(orig_img, [4*ones(1,9),3]);
%% Create the image

[sampled_image, mask] = SampleImage(orig_img, sampleFraction);
%T_observed = Img2Ket(sampled_image, dims);
T_observed = reshape(orig_img, [4*ones(1,9),3]);

mask_tensor = T_observed ~= 0;
omega = find(mask_tensor);     
missing_entries = find(~mask_tensor); 

fprintf('Original image size: %dx%dx%d\n', R, C, 3);
fprintf('Augmented tensor size: ');
fprintf('%d ', dims);
fprintf('\n');
fprintf('Sampling ratio: %.2f (%d%% missing)\n', SR, (1-SR)*100);
fprintf('Observed entries: %d, Missing entries: %d\n', numel(omega), numel(missing_entries));

%% Params TMac-TT

delta = zeros(1, N-1);
for k = 1:N-1
    delta(k) = min(prod(dims(1:k)), prod(dims(k+1:end)));
end
alpha = delta / sum(delta);

fprintf('Weights alpha_k: ');
fprintf('%.3f ', alpha);
fprintf('\n');

%% Initialize
X = T_observed; 

% Initialize missing pixels with the mean ----> otherwise wont optimize
% well
%mean_obs = mean(T_observed(omega));
%X(missing_entries) = mean_obs;

% Initialize U and V matrices properly
U = cell(1, N-1);
V = cell(1, N-1);

max_ranks = zeros(1, N-1);
for k = 1:N-1
    max_ranks(k) = min(prod(dims(1:k)), prod(dims(k+1:end)));
end

tt_ranks = min([3, 3, 3, 3, 3, 3, 3, 3, 3], max_ranks);

fprintf('TT ranks: ');
fprintf('%d ', tt_ranks);
fprintf('\n');

for k = 1:N-1
    mat_size = [prod(dims(1:k)), prod(dims(k+1:end))];
    Xk = reshape(X, mat_size);
    
    % Use simple SVD with rank adjustment
    [Uk, Sk, Vk] = safe_svd(Xk, tt_ranks(k));
    U{k} = Uk;
    V{k} = Sk * Vk';
end

%% TMac-TT Algorithm
fprintf('\nRunning TMac-TT algorithm...\n');
converged = false;
iter = 0;
errors = zeros(maxit, 1);
RSE_history = zeros(maxit, 1);

while ~converged && iter < maxit
    iter = iter + 1;
    
    previous_X = X;
    
    X_update = zeros(size(X));
    for k = 1:N-1
        mat_size = [prod(dims(1:k)), prod(dims(k+1:end))];
        Xk = reshape(X, mat_size);
        
        U{k} = Xk * V{k}';
        
        if size(U{k}, 2) < size(U{k}, 1)
            V{k} = (U{k}' * U{k} + 1e-8 * eye(size(U{k}, 2))) \ (U{k}' * Xk);
        else
            V{k} = U{k}' * Xk / (U{k} * U{k}' + 1e-8 * eye(size(U{k}, 1)));
        end
        
        Xk_approx = U{k} * V{k};
        
        Xk_tensor = reshape(Xk_approx, dims);
        X_update = X_update + alpha(k) * Xk_tensor;
    end
    X = X_update;
    
    X(omega) = T_observed(omega);
    
    diff_norm = norm(X(:) - previous_X(:)) / norm(previous_X(:) + eps);
    errors(iter) = diff_norm;
    
    %completed_img_current = Ket2Img(X, R, C);
    completed_img_current = reshape(X,[512,512,3]);

    RSE_history(iter) = norm(completed_img_current(:) - orig_img(:)) / norm(orig_img(:));
    
    if mod(iter, 20) == 0
        fprintf('Iteration %d: change=%.6f, RSE=%.4f\n', iter, diff_norm, RSE_history(iter));
    end
    
    if diff_norm < tol && iter > 10
        converged = true;
        fprintf('Converged at iteration %d\n', iter);
    end
end

errors = errors(1:iter);
RSE_history = RSE_history(1:iter);

%% Evaluation
completed_tensor = X;
%completed_img = Ket2Img(completed_tensor, R, C);
completed_img = reshape(completed_tensor,[512 512 3]);

RSE = norm(completed_img(:) - orig_img(:)) / norm(orig_img(:));
mse = mean((completed_img(:) - orig_img(:)).^2);
PSNR = 20 * log10(255 / sqrt(mse));

fprintf('\n=== Final Results ===\n');
fprintf('RSE: %.4f\n', RSE);
fprintf('PSNR: %.2f dB\n', PSNR);
fprintf('Final Iterations: %d\n', iter);

%% Display 
figure('Position', [100, 100, 1400, 400]);

% Original image
subplot(1, 5, 1);
imshow(orig_img / 255);
title('Original Image');

% Sampled image
subplot(1, 5, 2);
imshow(sampled_image / 255);
title(sprintf('Sampled (%.0f%% missing)', (1-SR)*100));

% Completed image
subplot(1, 5, 3);
imshow(completed_img / 255);
title(sprintf('TMac-TT Completed\nRSE: %.4f, PSNR: %.2f dB', RSE, PSNR));

% Error image
subplot(1, 5, 4);
error_img = abs(completed_img - orig_img);
imshow(error_img / max(error_img(:)));
title('Absolute Error');

% Difference from sampled image
subplot(1, 5, 5);
diff_from_sampled = abs(completed_img - sampled_image);
imshow(diff_from_sampled / max(diff_from_sampled(:)));
title('Diff from Sampled');

% Plot convergence
figure;
subplot(1,2,1);
semilogy(1:iter, errors, 'b-', 'LineWidth', 2);
xlabel('Iteration');
ylabel('Relative Change');
title('TMac-TT Convergence');
grid on;

subplot(1,2,2);
plot(1:iter, RSE_history, 'r-', 'LineWidth', 2);
xlabel('Iteration');
ylabel('RSE');
title('Reconstruction Error');
grid on;

fprintf('\nTMac-TT completion completed successfully!\n');

%% Safe SVD f
function [U, S, V] = safe_svd(A, k)
    [m, n] = size(A);
    
    k = min([k, m, n]);
    
    if k <= 0
        k = 1;
    end
    
    % Use standard SVD with economy decomposition
    [U, S, V] = svd(A, 'econ');
    
    % Truncate to k components
    U = U(:, 1:k);
    S = S(1:k, 1:k);
    V = V(:, 1:k);
  end
