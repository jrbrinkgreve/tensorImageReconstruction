%the main file from which subfunctions can be called
addpath('./funcs');

close all
%preprocessing
image_full = double(imread("4.2.07.tiff")) / 255; 


image = image_full / 255 ;%normalize 
shape = size(image);
imsize = [shape(1) shape(2)];


%sampling
sampleFraction = 0.3;%0.1, 0.3, 0.6
[sampled_image, mask] = SampleImage(image, sampleFraction);


figure
imshow(image_full);
title("Original image")
set(gca, 'FontSize', 24)

figure
imshow(sampled_image * 255);
title(sprintf("Subsampled image, sampleFraction = %.2f", sampleFraction))
set(gca, 'FontSize', 24)







%BASIC MATRIX COMPLETION ALG HERE

tau = 0.92;%1.48, 0.92, 0.466 for results as in report           %threshold step size, == tau in report * 303



%note: please do not touch these settings
delta = 1;     %stepsize: between 0 and 2 should work
eps = 1e-16;    %fro norm tolerance: past this point break loop
l = 1;         %not in use currently, can be related to size of svd later for speedup
kmax = 200; %10, 50, 1000    %max iterations, takes about 78 seconds for 1000 iterations (when using parallel channels)
k0 = 1;        %initial guess magnitude
opts_matrix = [delta eps tau l kmax k0]; %input options vector



[rec_img_matrix, rel_error_matrix] = SeparateMatrixCompletion(sampled_image, mask, opts_matrix);
relerr = norm(rec_img_matrix - image, 'fro') / norm(image, 'fro');




figure
imshow(rec_img_matrix * 255) % rescaling to 0-255 RGB dynamic range
title(sprintf('RecImg (relerr = %.4f)', relerr))
set(gca, 'FontSize', 24)


figure
semilogy(1:kmax, rel_error_matrix(:,1), color='red', LineWidth=3)
hold on
semilogy(1:kmax, rel_error_matrix(:,2),color='green', LineWidth=3)
semilogy(1:kmax, rel_error_matrix(:,3),color='blue', LineWidth=3)
title("Relative Error of Reconstruction to Measurement")
xlabel("Iteration number")
ylabel("RelError per channel")
legend("Red channel", "Green channel", "Blue channel")
set(gca, 'FontSize', 24)




%note: its kind of funny that the blue channel is worse: there are almost
%no blue pixels to reconstruct so the relative error on blue is also larger







%--------------------------------------------------------------------------------------------------------

%{

%MATRIX COMPLETION IN FOURIER SPACE

%note: please do not touch these settings
delta = 1.5;     %stepsize: between 0 and 2 should work
eps = 1e-10;    %fro norm tolerance: past this point break loop
tau = 0.2 * shape(1);   %to compensate for scaling of fft2(*) operation, just a suggested fix, not sure this is good
l = 1;         %not in use currently, should be related to size of svd later for speedup
kmax = 50;    %very slow 
k0 = 1;        %initial guess magnitude
opts_matrix = [delta eps tau l kmax k0]; %input options vector


[rec_img_matrix, rel_error_matrix] = FourierSeparateMatrixCompletion(sampled_image, mask, opts_matrix);
figure
imshow(rec_img_matrix * 255) % rescaling to 0-255 RGB dynamic range
title("Reconstructed image via separate channel matrix completion: fourier")

figure
semilogy(1:kmax, rel_error_matrix(:,1), color='red')
hold on
semilogy(1:kmax, rel_error_matrix(:,2),color='green')
semilogy(1:kmax, rel_error_matrix(:,3),color='blue')
title("Convergence of each individual channel, relative errors per channel")
xlabel("Iteration number")
ylabel("Relative frobenius norm of mismatch per channel")
legend("Red channel", "Green channel", "Blue channel")



%}


%--------------------------------------------------------------------------------------------------------


