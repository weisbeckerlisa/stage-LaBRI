clear; close all; clc;

addpath('utils');

I = imread('./data/335094.jpg');
E = imread('./data/335094_edge.png');
gaus=fspecial('gaussian',3);
I = imfilter(I,gaus,'replicate');

[h,w,z] = size(I);

mex  ./SuperpixelHierarchyMex.cpp -outdir ./
mex  ./utils/GetSuperpixels.cpp -outdir ./utils

close all
%%
% function model = SuperpixelHierarchyMex(image, edge, edge_weight, compactness)
% input:
% % image      : 3-channel image
% % edge        : 1-channel image
% % [edge_weight]: balance between edge and color feature 
% % [compactness]: [0,1]  (Rémi)
% output:
% % model: superpixel hierarchy
%%
tic; sh=SuperpixelHierarchyMex(I,E,0,1); toc

%get whatever you want
tic;
GetSuperpixels(sh,100);
toc;

figure, imagesc(sh.label)
%



%Script to select some scales
s_vect = 1000; %4000:-100:1; %[4000 3000 2000 1500 1000 500];
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

color1 = MeanColor(double(I),sh.label);
% GetSuperpixels(sh,1000); color2 = MeanColor(double(I),sh.label);
GetSuperpixels(sh,400); color3= MeanColor(double(I),sh.label);
% GetSuperpixels(sh,10); color4= MeanColor(double(I),sh.label);
% figure,imshow([color1,color2; color3,color4]);
figure, 
imshow(color3)

%%
% smooth boundary
tic; sh=SuperpixelHierarchyMex(I,E,0.9,0.1); toc
% get whatever you want
GetSuperpixels(sh,100); color1 = MeanColor(double(I),sh.label);
GetSuperpixels(sh,50); color2 = MeanColor(double(I),sh.label);
GetSuperpixels(sh,20); color3= MeanColor(double(I),sh.label);
GetSuperpixels(sh,10); color4= MeanColor(double(I),sh.label);
figure,imshow([color1,color2; color3,color4]);
%%

% compact superpixel
tic; N=1000; sh=SuperpixelHierarchyMex(I,E,0.9,0); toc
% get whatever you want
GetSuperpixels(sh,1000); color1 = MeanColor(double(I),sh.label);
GetSuperpixels(sh,800); color2 = MeanColor(double(I),sh.label); 
GetSuperpixels(sh,400); color3= MeanColor(double(I),sh.label); 
GetSuperpixels(sh,100); color4= MeanColor(double(I),sh.label); 
figure,imshow([color1,color2; color3,color4]);
