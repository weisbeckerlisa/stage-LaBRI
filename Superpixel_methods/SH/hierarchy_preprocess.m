clear; close all; clc;

addpath('utils');

% Setup folders
root_folder = 'data/accv/preprocessed_data';
root_edges = 'data/edges/preprocessed_data';
image_folders = {'ref', 'target'};
result_folders = 'labels/preprocessed_data';

% Compilation
% mex  ./SuperpixelHierarchyMex.cpp -outdir ./
% mex  ./utils/GetSuperpixels.cpp -outdir ./utils


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
            I = cat(3, I, I, I);
        end

        tic; sh=SuperpixelHierarchyMex(I,E,0,1); toc
        sp = size(I,1)/4;
        GetSuperpixels(sh, sp);
        S_SH = sh.label;
        [~, name, ~] = fileparts(image_files(image_idx).name);
        mat_filename = fullfile(result_folder, name);
        save(mat_filename, 'S_SH');
    end
end
