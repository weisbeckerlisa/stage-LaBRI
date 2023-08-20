
% Initialisation
clear; close all; clc;

% Changer ces chemins selon votre configuration
imagesFolder = 'data/cocoval';
labelsFolder = 'data/cocoval_mat';
gtFolder = 'data/cocoval_gt';
spFolder = 'data/cocoval_bsds';
outputFolder = 'data/new_seg';

% Obtenir les noms de tous les fichiers d'image
imageFiles = dir(fullfile(imagesFolder, '*.jpg'));
labelFiles = dir(fullfile(labelsFolder, '*.mat'));
gtFiles = dir(fullfile(gtFolder, '*.png'));

img_idx = 22;

[~, name, ~] = fileparts(imageFiles(img_idx).name);

% Obtenir le nom d'une image et son label
imageFile = fullfile(imagesFolder, [name, '.jpg']);
labelFile = fullfile(labelsFolder, name);
gtFile = fullfile(gtFolder, [name, '.png']);

% Charger l'image et la segmentation d'origine
img = imread(imageFile);
labelData = load(labelFile); 
labelImg = labelData.label +1;
gtImg = imread(gtFile);

% Exécuter la segmentation par superpixels pour différentes quantités de superpixels
SP_numbers = [25, 50, 100, 150, 325, 600]; % nombre de superpixels
for k = 1:numel(SP_numbers)
    SP = SLIC(single(img)/255, SP_numbers(k), 10, 10); 
    % Créer une nouvelle image de segmentation
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