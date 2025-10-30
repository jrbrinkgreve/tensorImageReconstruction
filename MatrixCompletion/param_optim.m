%parameter optimization script:
%the main file from which subfunctions can be called
addpath('./funcs');

close all
%preprocessing

disp("starting optim loop")
image_full = double(imread("4.2.07.tiff")); 


image = image_full/255;%normalize 
shape = size(image);
imsize = [shape(1) shape(2)];


%sampling
sampleFraction = 0.3;%0.1, 0.3, 0.6
[sampled_image, mask] = SampleImage(image, sampleFraction);





num_optimpoints = 16;
tau_array = linspace(200,400, num_optimpoints);
relerr_array = zeros(size(tau_array));


parfor i = 1:length(tau_array)

tau = tau_array(i);
i





%note: please do not touch these settings
delta = 1;     %stepsize: between 0 and 2 should work
eps = 1e-16;    %fro norm tolerance: past this point break loop
%tau = 0.92%0.05, 0.2, 0.3;     %threshold step size

l = 1;         %not in use currently, should be related to size of svd later for speedup
kmax = 200; %10, 50, 1000    %max iterations, takes about 78 seconds for 1000 iterations (parallel channels)
k0 = 1;        %initial guess magnitude
opts_matrix = [delta eps tau l kmax k0]; %input options vector



[rec_img_matrix, rel_error_matrix] = SeparateMatrixCompletion(sampled_image, mask, opts_matrix);
relerr = norm(rec_img_matrix - image, 'fro') / norm(image, 'fro');











relerr_array(i) = relerr;

end




% Find best cutoff (minimum RSE)
[bestRSE, idx] = min(relerr_array);
bestCutoff = tau_array(idx);

% Create figure
figure;
plot(tau_array, relerr_array, '-o', 'LineWidth', 1.5, 'MarkerSize', 6);
hold on;

% Highlight best cutoff
plot(bestCutoff, bestRSE, 'r*', 'MarkerSize', 10, 'LineWidth', 1.5);

% Labels and title
xlabel('Cutoff', 'FontSize', 12);
ylabel('RSE (relative reconstruction error)', 'FontSize', 12);
title('Cutoff sweep: RSE vs cutoff', 'FontSize', 13);

% Annotation (legend-style)
legend({'RSE', sprintf('Best cutoff = %.5f', bestCutoff)}, 'Location', 'northwest');

% Grid and formatting
grid on;
set(gca, 'FontSize', 11, 'Box', 'on', 'XMinorGrid', 'off', 'YMinorGrid', 'off');
xlim([min(tau_array) max(tau_array)]);

% Optional: mark point text
text(bestCutoff*1.1, bestRSE, sprintf('  min RSE = %.3f', bestRSE), ...
    'Color', 'r', 'FontSize', 10);

hold off;

