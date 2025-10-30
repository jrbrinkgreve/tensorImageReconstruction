function [sampled_image, mask] = SampleImage(image, sampleFraction)
%{
FUNCTION
this function takes as input an image and a fraction, and outputs a
randomly subsampled image. 

IN
image: N x M (x3) image (RGB)
samplefraction: double between 0 and 1


OUT
image: N x M (x3) image (RGB) with missing pixels

NOTES
over the colour dimension no different samplings are taken, as you'd expect
%}

shape = size(image);
imsize = [shape(1) shape(2)];
numPixels = prod(imsize);

mask_indices = randsample(numPixels, round(sampleFraction * numPixels) );
mask = zeros(imsize);
mask(mask_indices) = 1;

sampled_image = image .* mask;