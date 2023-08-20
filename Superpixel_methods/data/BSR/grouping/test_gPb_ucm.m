%% Compute globalPb and hierarchical segmentation for an example image.

addpath(fullfile(pwd,'lib'));

%% 1. compute globalPb on a small image to test mex files
clear all; close all; clc;

imgFile = 'data/101087_small.jpg';
imgFile = 'peppers.png';
outFile = 'data/101087_small_gPb.mat';

figure, imagesc(imread(imgFile));

tic;
gPb_orient = globalPb(imgFile, outFile);
delete(outFile);
toc;

figure; imshow(max(gPb_orient,[],3)); colormap(jet);

tic;
ucm = contours2ucm(gPb_orient, 'imageSize');
toc;
figure;imshow(ucm);
%%

lab_map = regions_from_closed_contours(ucm, 'orders_fill')



%%
figure,
imagesc(lab_map)
