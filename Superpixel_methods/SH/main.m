clear; close all; clc;

addpath('utils');

I = imread('./data/335094.jpg');
E = imread('./data/335094_edge.png');
gaus=fspecial('gaussian',3);
I = imfilter(I,gaus,'replicate');

[h,w,z] = size(I);

mex -O CXXFLAGS="\$CXXFLAGS -O3 -Wall -lm" CXXOPTIMFLAGS="-O3" ./SuperpixelHierarchyMex.cpp -outdir ./
mex -O CXXFLAGS="\$CXXFLAGS -O3 -Wall -lm" CXXOPTIMFLAGS="-O3" ./utils/GetSuperpixels.cpp -outdir ./utils

%%
% function model = SuperpixelHierarchyMex(image, edge, edge_weight, compactness)
% input:
% % image      : 3-channel image
% % edge        : 1-channel image
% % [edge_weight]: balance between edge and color feature (default: 4)
% % [compactness]: [0,1] (default 4) (Rémi)
% output:
% % model: superpixel hierarchy

tic; sh=SuperpixelHierarchyMex(I,E*0,4,4); toc

%%
%Script to select some scales
s_vect = [4000 3000 2000 1500 1000 500];
GetSuperpixels(sh,s_vect(1));
S = 0 + sh.label; %To force copy instead of pointer (???!!)

classif = 0:s_vect(1)-1;
for k=2:length(s_vect)
    k
    GetSuperpixels(sh,s_vect(k));
    Sj = sh.label;
    for i=1:h
        for j=1:w
            classif(k,S(i,j)+1) = Sj(i,j);
        end
    end
end

figure, 
imagesc(sh.label);

