function rec_img = TensorCompletion(sampled_image, mask, opts)
%image reconstruction via individual matrix completion
shape = size(sampled_image);
imsize = [shape(1) shape(2)];
numPixels = prod(imsize);


%approach: use TT to get smaller submatrices, then lower the rank of these
%submatrices




%512 x 512 x 3 == 786432 == 4^9 * 3
tsi = reshape(sampled_image, [linspace(4, 4, 9), 3]);  %Tensorized_Sampled_Image



%get an exact TT of the sampled image
tt_tsi = TT_SVD(tsi, 1e-16);

%------------------------------------------------------------------------------------------------------------
%do some processing to reconstruct the original image in its TT form:
%some kind of rank minimization algorithm on the TT...













%------------------------------------------------------------------------------------------------------------

%reconstruct the tensor from the TT
rec_tt_tsi = TT_reconstruct(tt_tsi);


%reshape the tensor back into the image
rec_img = reshape(rec_tt_tsi, shape);















end