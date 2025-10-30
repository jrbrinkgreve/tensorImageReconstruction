% Simple Low Rank Tensor Completion
image = imread("4.2.07.tiff");
[x, y, c] = size(image);

% the mask  can be selected here
%mask = rand(x, y) > 0.9; %for sampling rate 0.1
%mask = rand(x, y) > 0.7; %for sampling rate 0.3
mask = rand(x, y) > 0.4; %for sampling rate 0.6

th = 0.001; % the threshold


image_noisy = uint8(zeros(x, y, c));  %construct the masked image
for k=1:c
    for i=1:x
        for j = 1:y
            if mask(i,j) == 1
                image_noisy(i,j,k) = image(i,j,k);
            end
        end
    end
end




orig = double(image);
img = double(image_noisy);
norm_orig = norm(orig(:));

[x, y, c] = size(img);
X = img;
shape_inf = [x, y, c];


weights = [512,512,3];  % weigths are assigned based on matricization ranks
 
weights = weights/sum(weights);


M = cell(1, c);
norm_error = 10000;
idx = 0;
while true
    idx = idx + 1
    summed = zeros(x,y,c);
    parfor n = 1:c
        % Mode-n unfolding
        order = [n, 1:n-1, n+1:ndims(X)]; 
        X_perm = permute(X, order);
        X_n = reshape(X_perm, size(X, n), []);
    
        % SVD and soft threshold
        [U, S, V] = svd(X_n, 'econ');

        tau = thresholder(S,th);
        S_new = wthresh(S, 's', tau);
        M{n} = U * S_new * V';
    
        % Fold back to tensor form
        X_rec = reshape(M{n}, size(X_perm));
        X_rec = ipermute(X_rec, order);   
        summed = summed +  X_rec*weights(n);
    end
    
    % keep observed pixels
    X = summed;
    X(repmat(mask, [1,1,c])) = img(repmat(mask, [1,1,c]));
    err_X   = norm(orig(:) - X(:)) / norm_orig;   %fint the frobenius norm error
 
    if abs(norm_error-err_X) < 1e-4
        break;
    end
    norm_error = err_X;
    

end
error_stored(i)= err_X;


figure;

imshow(uint8(X));
title(sprintf('Recovered Image (Error = %.4f)', err_X));



function tx = thresholder(S,th)    %function for finding the soft thresholding value based on ratio threshold.
    val = 10;
    for u=1:size(S,1)
        if S(u,u)/S(1,1) < th
            val = S(u,u);
            break;
        end
    end
    tx = val;
end

