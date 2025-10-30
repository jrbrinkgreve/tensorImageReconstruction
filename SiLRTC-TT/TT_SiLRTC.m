image = imread("4.2.07.tiff");
[x, y, c] = size(image);
image_noisy = uint8(zeros(x, y, c));


%chose the sampling rate and the corrresponding threshold here
%mask = rand(x, y) > 0.9; %for sampling rate 0.1
%th = 0.002

%mask = rand(x, y) > 0.7; %for sampling rate 0.3
%th = 0.001

mask = rand(x, y) > 0.4; %for sampling rate 0.6
th = 0.001;


for k=1:c    %generate masked image
    for i=1:x
        for j = 1:y
            if mask(i,j) == 1
                image_noisy(i,j,k) = image(i,j,k);
            end
        end
    end
end

img = double(image_noisy);
orig = double(image);
norm_orig = norm(orig(:)); 

[x, y, c] = size(img);
X_tt = img;   %initialize image 


M = cell(1, c);

norm_error = 10000; %just an initialization for error calc so it doesnt crash
shape_inf = [4,4,4,4,4,4,4,4,4,3];  %shape of the TT
weights = zeros(1,length(shape_inf)-1);

for i = 1:(length(shape_inf)-1)   %weight assignment based on max ranks of the TT
    w1 = prod(shape_inf(1:i));
    w2 = prod(shape_inf(i+1:end));

    weights(i) = min(w1,w2);

end

weights = weights/sum(weights);



idx=0;
while true
    idx = idx + 1
    summed = zeros(shape_inf);
    tensor_10 = reshape(X_tt,shape_inf);  %tensorize it
    a = 1;
    for i = 1:(ndims(tensor_10)-1)
        order =[1:i i+1:ndims(tensor_10)];  %mode(1,2..n) matricization
        a = a*shape_inf(i);
        t_perm = permute(tensor_10, order); 
        X_n = reshape(t_perm, a, []);
    
        [U, S, V] = svd(X_n, 'econ'); %svd and thresholding
        tau = thresholder(S,th);
        S_new = max(S- tau, 0);
        M{i} = U * S_new * V';

        X_rec = reshape(M{i}, shape_inf); %reconstruct the tensor and average across different matrices using weights
        X_rec = ipermute(X_rec, order);
        summed = summed +  X_rec*weights(i);
    end
    X_recovered = reshape(summed, 512, 512, 3);
    X_recovered(repmat(mask, [1,1,c])) = img(repmat(mask, [1,1,c])); %keep observed pixels
    X_tt = X_recovered;

    err_Xtt  = norm(orig(:) - X_tt(:)) / norm_orig; %frobenius error

    if abs(norm_error-err_Xtt) < 1e-4
        break;
    end

    norm_error = err_Xtt;

end



figure;

imshow(uint8(X_tt));
title(sprintf('Recovered Image (Error = %.4f)', err_Xtt));





function tx = thresholder(S,th)
    val = 10;
    for u=1:size(S,1)
        if S(u,u)/S(1,1) < th
            val = S(u,u);
            break;
        end
    end
    tx = val;
end

