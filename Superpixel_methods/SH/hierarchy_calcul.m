clear; close all; clc;

addpath('utils');

% Setup folders
root_folder = 'data/accv';
root_edges = 'data/edges';
image_folders = {'ref', 'target'};
result_folders = 'results';

% Compilation
% mex  ./SuperpixelHierarchyMex.cpp -outdir ./
% mex  ./utils/GetSuperpixels.cpp -outdir ./utils

SP_range = [10, 20, 50, 100, 200, 400, 700, 1000];


for folder_idx = 1:length(image_folders)
    image_folder = fullfile(root_folder, image_folders{folder_idx});
    edges_folder = fullfile(root_edges, image_folders{folder_idx});
    result_folder = fullfile(result_folders, image_folders{folder_idx});
    image_files = dir(fullfile(image_folder, '*.png'));
    edges_files = dir(fullfile(edges_folder, '*.png'));

    for image_idx = 1:length(image_files)
        image_file = fullfile(image_folder, image_files(image_idx).name);
        edges_file = fullfile(edges_folder, edges_files(image_idx).name);
        I = imread(image_file);
        E = imread(edges_file);

        gaus=fspecial('gaussian',3);

        I = imfilter(I, gaus, 'replicate');
        if size(I, 3) == 1
            % L'image est en niveaux de gris, convertir en RGB
            I = cat(3, I, I, I);
        end

        tic; sh=SuperpixelHierarchyMex(I,E,0,1); toc

        for sp = SP_range
            GetSuperpixels(sh, sp);
            S_SH = sh.label;
            color = MeanColor(double(I), S_SH);
            imwrite(color, fullfile(result_folder, 'meanColor', [image_files(image_idx).name,'_', num2str(sp), '.png']));
            borders = compute_border(S_SH, I);
            imwrite(double(I)/255.*repmat(~borders, [1 1 3]), fullfile(result_folder, 'superpixels', [image_files(image_idx).name,'_', num2str(sp), '.png']));
        end
    end
end

