clear
close all 
clc

dims = [repmat(4,1,9),3];
N = numel(dims);
sample = 0.1;

img = double(imread("4.2.07.tiff"));

X_orig = Img2Ket(img,dims);

obs = randsample(prod(dims),round(sample*prod(dims)));
[ind,~] = sort(obs);

Xn = X_orig(ind);
mask = zeros(dims);
mask(ind) = 1;
%mask = logical(mask);
img_obs = mask.*X_orig;
M = zeros(dims);
M(ind) = Xn;
for k = 2:length(ind)
    inter = ind(k)-ind(k-1);
    if inter > 1
        x0 = ind(k-1)+1;
        y0 = ind(k)-1;
        meank = (Xn(k) + Xn(k-1))/2;
        M(x0:y0) = meank;        
    end
end
if length(ind)<length(M(:))
    x0 = ind(end)+1;
    y0 = length(M(:));
    meank = Xn(end);
    M(x0:y0) = meank;
end
cutoff = 0.002;
tolerance = 10^(-8);
maxiter = 800;

[~,rank] = TT_SVD_Rank(X_orig,cutoff);

%% Weights
lambda = zeros(1,N-1);
IL = dims(1);
for k = 1:N-1
    IR = prod(dims(k+1:end));
    lambda(k) = min(IL,IR);
    IL = IL*dims(k+1);
end
lambda = lambda/(sum(lambda));

%% Init M


%% Optimization
tic;


X0 = cell(1,N-1);Y0 = cell(1,N-1);
dimL = zeros(1,N-1);
dimR = zeros(1,N-1);
IL = 1;
for k = 1:N-1
    dimL(k) = IL*dims(k);
    dimR(k) = prod(dims)/dimL(k);
    X0{k} = randn(dimL(k),rank(k));
    Y0{k} = randn(rank(k),dimR(k));

    IL = dimL(k);
end

X = X0;Y = Y0; 
normM = norm(Xn(:),'fro');
Xsq = cell(1,N-1);
k = 1;
relerr = [];
maxit = maxiter;
relerr(1) = 1;

fprintf('iter: RSE  \n');

while relerr(k) > tolerance
    k = k+1;
    Mcur = M;


    for n = 1:N-1
        Mn = reshape(M,[size(X{n},1) size(Y{n},2)]);
        X{n} = Mn*Y{n}';
        Xsq{n} = X{n}'*X{n};
        Y{n} = pinv(Xsq{n})*X{n}'*Mn;
    end
    Mn = X{1}*Y{1};
    M = lambda(1)*Mn;
    M = reshape(M,dims);
    for n = 2:N-1
        Mn = X{n}*Y{n};
        Mn = reshape(Mn,dims);
        M = M+lambda(n)*Mn;
    end

    M(ind) = Xn;

    relerr(k) = abs(norm(M(:)-Mcur(:)) / normM);
    if mod(k, 100) == 0
        R = 512; C = 512;
        Img = Ket2Img(M, R, C);
        %Img0 = reshape(X_orig, [R, C, 3]);
        Img0 = img;
        fprintf('  %d:  %f \n',k-1,norm(Img-Img0,'fro')/norm(Img,'fro'));
    end

    if k > maxit || (k > 2 && relerr(k) > relerr(k-1))
        break
    end
end
timeTC = toc;

R = 512; C = 512;

Img_recovered = Ket2Img(M, R, C);
%Img_original = reshape(X_orig, [R, C, 3]);
Img_original = img;
h3 = figure();
set(h3,'Position',[700 150 400 350]);
imagesc(uint8(Img_recovered));
title('Reconstructed image');
h4 = figure();
%set(h4,'Position',[200 150 1200 400]);


% % Plot convergence
% h5 = figure();
% %set(h5,'Position',[200 600 600 400]);
% if iscell(err)
%     plot(err{1});
%     hold on;
%     plot(tol*ones(1,length(err{1})), 'r--');
% else
%     plot(err);
%     hold on;
%     plot(tol*ones(1,length(err)), 'r--');
% end
% grid on;
% set(gca,'YScale','log');
% title('Convergence History');
% xlabel('Iteration');
% ylabel('Relative Error');
% legend('Relative Error', 'Tolerance');
% 
% fprintf('\nFinal Results:\n');
% fprintf('RSE: %.6f\n', RSEmin);
% fprintf('Time: %.2f seconds\n', timeTS);
% fprintf('Threshold: %.4f\n', muf);
% fprintf('Iterations: %d\n', length(err(1)));