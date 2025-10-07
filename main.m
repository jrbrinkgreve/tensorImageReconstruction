%the main file from which subfunctions can be called

%preprocessing
image_full = double(imread("4.2.07.tiff")) / 255; 

image = image_full / 255;%normalize between 0 and 1 instead of 0 and 255
shape = size(image);
imsize = [shape(1) shape(2)];


%sampling
sampleFraction = 0.5;
[sampled_image, mask] = SampleImage(image, sampleFraction);



%separate matrix completion algorithm here...
opts_matrix = []; %input options vector
SeparateMatrixCompletion(sampled_image, mask, opts_matrix);



%tensor completion algorithm here
opts_tensor = [];
TensorCompletion(sampled_image, mask, opts_tensor);





%postprocessing / data visualization / convergence plots etc