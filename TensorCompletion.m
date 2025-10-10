function rec_img = TensorCompletion(sampled_image, mask, opts)
%image reconstruction via individual matrix completion
shape = size(sampled_image);
imsize = [shape(1) shape(2)];
numPixels = prod(imsize);



rec_img = zeros(shape);

%approach: use TT to get smaller submatrices, then lower the rank of these
%submatrices

%512 x 512 x 3 == 786432 == 4^9 * 3
tsi = reshape(sampled_image, [linspace(4, 4, 9), 3]);  %Tensorized_Sampled_Image


tt_tsi = TT_SVD(tsi, 1e-16); % to get an exact TT of the sampled image



%then build some kind of algorithm here that lowers the rank of the TT to
%reconstruct the image









end