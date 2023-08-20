
% Initialisation
clear; close all; clc;

imagesFolder = 'data/cocoval';
labelsFolder = 'data/cocoval_mat';
gtFolder = 'data/cocoval_gt';
spFolder = 'data/cocoval_500k';
outputFolder = 'data/new_seg';

imageFiles = dir(fullfile(imagesFolder, '*.jpg'));
labelFiles = dir(fullfile(labelsFolder, '*.mat'));
spFiles = dir(fullfile(spFolder, '*.mat'));
gtFiles = dir(fullfile(gtFolder, '*.png'));

img_idx = 22;

[~, name, ~] = fileparts(imageFiles(img_idx).name);

imageFile = fullfile(imagesFolder, [name, '.jpg']);
labelFile = fullfile(labelsFolder, name);
gtFile = fullfile(gtFolder, [name, '.png']);

% Charger l'image et la segmentation d'origine
img = imread(imageFile);
labelData = load(labelFile); 
labelImg = labelData.label +1;
gtImg = imread(gtFile);

SP_numbers = [25, 50, 100, 150, 325, 600]; % nombre de superpixels
for k = 1:numel(SP_numbers)
    SP = load(fullfile(spFolder, [name, '_', num2str(SP_numbers(k)), '_label.mat'])).label;
    newSegmentation = zeros(size(labelImg));

    % Attribuer les labels de classe
    for j = 0:max(SP, [], 'all')
        mask = SP == j;
        labels_in_superpixel = labelImg(mask);
        modeLabel = mode(labels_in_superpixel(:));
        newSegmentation(mask) = modeLabel;
    end

    figure; 
    subplot(2, 2, 1); 
    imagesc(label2rgb(labelImg));
    title('Groundtruth');
    subplot(2, 2, 2); 
    imagesc(label2rgb(newSegmentation)); 
    title(['Affinage - Superpixels: ', num2str(SP_numbers(k))]);
    B_orig = compute_borders_rgb(labelImg, img, [0,1,0]);
    subplot(2, 2, 3)
    imagesc(double(img)/255.*~B_orig);
    B_new = compute_borders_rgb(newSegmentation, img, [0,1,0]);
    subplot(2, 2, 4)
    imagesc(double(img)/255.*~B_new);

end
