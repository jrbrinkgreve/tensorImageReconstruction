% Simple Low Rank Tensor Completion

img = double(image_noisy);
[x, y, c] = size(img);
X = img;

tau = 3000;
alpha = 1/c;

M = cell(1, c);
summed = zeros(x,y,c);

for count=1:39
    for n = 1:c
        % Mode-n unfolding
        order = [n, 1:n-1, n+1:ndims(X)];
        X_perm = permute(X, order);
        X_n = reshape(X_perm, size(X, n), []);
    
        % SVD and soft threshold
        [U, S, V] = svd(X_n, 'econ');
        S_new = wthresh(S, 's', tau);
        M{n} = U * S_new * V';
    
        % Fold back to tensor form
        X_rec = reshape(M{n}, size(X_perm));
        X_rec = ipermute(X_rec, order);
    
    
        summed = summed +  X_rec/60;
    end
    
    % Data consistency (keep observed pixels)
    for i = 1:x
        for j = 1:y
            if mask(i,j) == 1
                summed(i,j,:) = img(i,j,:);
            end
        end
    end
    
    X = summed;
end

figure;

subplot(1,2,1);            % 1 row, 2 columns, first plot
imshow(image_noisy);
title('Noisy / Sparse Image');

subplot(1,2,2);            % second plot
imshow(uint8(X));
title('Recovered Image');

