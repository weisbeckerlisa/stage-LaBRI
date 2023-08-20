clear; close all; clc

%Compilation
% mex -O CFLAGS="\$CFLAGS -Wall -Wextra -W -std=c99" ./SLIC/SLIC.c -outdir ./SLIC  
% mex -O CFLAGS="\$CFLAGS -Wall -Wextra -W -std=c99" ./LSC/LSC_original.cpp -outdir ./LSC  
% mex -O CFLAGS="\$CFLAGS -Wall -Wextra -W -std=c99" ./SCALP/SCALP_mex.cpp -outdir ./SCALP  
% mex -O CFLAGS="\$CFLAGS -Wall -Wextra -W -std=c99" ./SH/SuperpixelHierarchyMex.cpp -outdir ./SH  

addpath(genpath('.'));
data_path = './data';

img_name = sprintf('%s/test_img.jpg',data_path); 
I = imread(img_name);
%I = imnoise(I,'gaussian',0,0.001); % Ajoute un bruit gaussien
[h,w,z] = size(I);

% Parameters
SP_nbr = 25;    % Superpixel number

% Contour detection (for SCALP)
C = double(imread(sprintf('%s/test_img_contour.png' ,data_path)));
C = C/max(C(:));

%%
tic;
S_slic  = SLIC(single(I)/255, SP_nbr, 10, 10); 
toc;
tic;
S_lsc   = LSC_original(uint8(I), SP_nbr, 0.075); 
toc;
tic;
S_scalp = SCALP_mex(uint8(I), SP_nbr, 0.075, single(C)); %S = SCALP_mex(I,SP_nbr,0.075);  %without contour prior
toc;
tic;
tic;
sh = SuperpixelHierarchyMex(uint8(I), uint8(C*255), 10, 1000); GetSuperpixels(sh, SP_nbr); S_SH = sh.label;
toc;
tic;
data_SSN = load('data/test_img_25_label.mat');
S_SSN = data_SSN.labels;
toc;

B_slic  = compute_border(S_slic, I);
B_lsc   = compute_border(S_lsc, I);
B_scalp = compute_border(S_scalp, I);
B_SH = compute_border(S_SH, I);
B_SSN = compute_border(S_SSN, I);

figure,
subplot 231
imagesc(double(I)/255.*repmat(~B_slic,[1 1 3])); title('SLIC');
subplot 232
imagesc(double(I)/255.*repmat(~B_lsc,[1 1 3]));  title('LSC');
subplot 233
imagesc(double(I)/255.*repmat(~B_scalp,[1 1 3])); title('SCALP');
subplot 234
imagesc(double(I)/255.*repmat(~B_SH,[1 1 3])); title('SH');
subplot 235
imagesc(double(I)/255.*repmat(~B_SSN,[1 1 3])); title('SSN');

which asa_metric

%Display & metric

gt2 = imread('data/test_img_gt.png');

if (exist('utils_sp'))
    main_fct_sp(I,gt2,S_SSN);
end


