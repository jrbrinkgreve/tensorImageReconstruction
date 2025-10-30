function [rec_img, rel_error] = SeparateMatrixCompletion(sampled_image, mask, opts)


shape = size(sampled_image);
imsize = [shape(1) shape(2)];
channels = shape(3);
numPixels = prod(imsize);

delta = opts(1);
eps = opts(2);
tau = opts(3);
l = opts(4);
kmax = opts(5);
k0 = opts(6);




%initializations
X = zeros(shape);
sampled_image_norm = zeros(1,1,3);
for i = 1:3
sampled_image_norm(:,:,i) = norm(sampled_image(:,:,i), 'fro');
end
rel_error = double(zeros(kmax, channels));

%loop over separate channels
parfor c = 1:channels
Y = k0 * delta * sampled_image(:,:,c);
    

%SVT iterations
    for k = 1:kmax
        %if mod(k, 100) == 0    %progress tracker
        %    disp(k)
        %end
       [U,S,V] = svd(Y);   %do the SVD
        S = max(S-tau, 0);  %threshold
        X(:,:,c) = U*S*V';   %construct low rank approximation
   
        rel_error(k, c) = norm(mask.*(X(:,:,c)-sampled_image(:,:,c)), 'fro') / sampled_image_norm(:,:,c);   %print error per channel
        
        if rel_error(k, c) < eps
            break
        end

        %gradient step
        Y = Y + delta * mask.* (sampled_image(:,:,c) - X(:,:,c));
        
    end
end

%return image
rec_img = X;


end