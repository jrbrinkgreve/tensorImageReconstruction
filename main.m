%the main file from which subfunctions can be called

%preprocessing
image_full = double(imread("4.2.07.tiff")) / 255; 

image = image_full / 255;%normalize between 0 and 1 instead of 0 and 255
shape = size(image);
imsize = [shape(1) shape(2)];


%sampling
sampleFraction = 0.5;
[sampled_image, mask] = SampleImage(image, sampleFraction);
figure
imshow(image_full);
title("Original image")

figure
imshow(sampled_image * 255);
title("Subsampled image")







%separate matrix completion algorithm here...

%note: please do not touch these settings
delta = 1.5;     %stepsize: between 0 and 2 should work
eps = 1e-10;    %fro norm tolerance: past this point break loop
tau = 0.2;     %threshold step size
l = 1;         %not in use currently, should be related to size of svd later for speedup
kmax = 1000;    %max iterations, takes about 78 seconds for 1000 iterations (parallel channels)
k0 = 1;        %initial guess magnitude
opts_matrix = [delta eps tau l kmax k0]; %input options vector


[rec_img_matrix, rel_error_matrix] = SeparateMatrixCompletion(sampled_image, mask, opts_matrix);
figure
imshow(rec_img_matrix * 255) % rescaling to 0-255 RGB dynamic range
title("Reconstructed image via separate channel matrix completion")

figure
semilogy(1:kmax, rel_error_matrix(:,1), color='red')
hold on
semilogy(1:kmax, rel_error_matrix(:,2),color='green')
semilogy(1:kmax, rel_error_matrix(:,3),color='blue')
title("Convergence of each individual channel, relative errors per channel")
xlabel("Iteration number")
ylabel("Relative frobenius norm of mismatch per channel")
legend("Red channel", "Green channel", "Blue channel")


%note: its kind of funny that the blue channel is worse: there are almost
%no blue pixels to reconstruct so the relative error on blue is also larger





%tensor completion algorithm here
opts_tensor = [];
TensorCompletion(sampled_image, mask, opts_tensor);





%postprocessing / data visualization / convergence plots etc