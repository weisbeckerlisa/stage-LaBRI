% Initialisation
clear; close all; clc;

% Compilation
% mex -O CFLAGS="\$CFLAGS -Wall -Wextra -W -std=c99" ./SLIC/SLIC.c -outdir ./SLIC  
% mex -O CFLAGS="\$CFLAGS -Wall -Wextra -W -std=c99" ./LSC/LSC_original.cpp -outdir ./LSC  
% mex -O CFLAGS="\$CFLAGS -Wall -Wextra -W -std=c99" ./SCALP/SCALP_mex.cpp -outdir ./SCALP  
% mex -O CFLAGS="\$CFLAGS -Wall -Wextra -W -std=c99" ./SH/SuperpixelHierarchyMex.cpp -outdir ./SH  

% Ajouter des dossiers au chemin
addpath(genpath('.'));

% Dossiers de données
root_folder = 'data/BSR/BSDS500/data';
image_folders = {'images/test'}%, 'images/train', 'images/val'};
gt_folders = {'groundtruth/test'}%, 'groundtruth/train', 'groundtruth/val'};
edge_folder = 'edges';
ssn_result_folder = 'data/ssn_result';

% Initialiser les métriques pour chaque méthode
ASA_metrics_slic = [];
ASA_metrics_lsc = [];
ASA_metrics_scalp = [];
ASA_metrics_sh = [];
ASA_metrics_ssn = [];

EV_metrics_slic = [];
EV_metrics_lsc = [];
EV_metrics_scalp = [];
EV_metrics_sh = [];
EV_metrics_ssn = [];

GR_metrics_slic = [];
GR_metrics_lsc = [];
GR_metrics_scalp = [];
GR_metrics_sh = [];
GR_metrics_ssn = [];

np_range = [1000, 800, 600, 500, 425, 350, 275, 200, 150, 100, 50, 25, 10];

% Parcourir chaque image et calculer les métriques
for sp_nbr = np_range
    for folder_idx = 1:length(image_folders)
        image_folder = fullfile(root_folder, image_folders{folder_idx});
        gt_folder = fullfile(root_folder, gt_folders{folder_idx});
        
        image_files = dir(fullfile(image_folder, '*.jpg'));
       
        % Initialize per folder metric storage
        ASA_folder_metrics_slic = [];
        ASA_folder_metrics_lsc = [];
        ASA_folder_metrics_scalp = [];
        ASA_folder_metrics_sh = [];
        ASA_folder_metrics_ssn = [];
        
        EV_folder_metrics_slic = [];
        EV_folder_metrics_lsc = [];
        EV_folder_metrics_scalp = [];
        EV_folder_metrics_sh = [];
        EV_folder_metrics_ssn = [];
        
        GR_folder_metrics_slic = [];
        GR_folder_metrics_lsc = [];
        GR_folder_metrics_scalp = [];
        GR_folder_metrics_sh = [];
        GR_folder_metrics_ssn = [];
        
        for image_idx = 1:length(image_files)
            % Lire l'image et calculer les contours
            image_files(image_idx).name
            image_file = fullfile(image_folder, image_files(image_idx).name);
            I = imread(image_file);

            I = double(I);
            
            % Lire les fichiers ground truth
            [~, name, ~] = fileparts(image_files(image_idx).name);
            gt_file = fullfile(gt_folder, [name, '.mat']);
            gt_cell = load(gt_file);
            
            % Lire l'image de contour
            edge_file = fullfile(edge_folder, [name, '_contour.png']);
            C = double(imread(edge_file));
            C = C/max(C(:));
            
            % Calculer les superpixels et les métriques pour chaque méthode
            S_slic  = SLIC(single(I)/255, sp_nbr, 10, 10); 
            S_lsc   = LSC_original(uint8(I), sp_nbr, 0.075);
            S_scalp = SCALP_mex(uint8(I), sp_nbr, 0.075, single(C));
            sh = SuperpixelHierarchyMex(uint8(I), uint8(C*255), 10, 1000);
            GetSuperpixels(sh, sp_nbr);
            S_SH = sh.label;
            
            % Lire résultat pré-calculé de SSN
            S_SSN = load(fullfile(ssn_result_folder, [name, '_', num2str(sp_nbr), '_label.mat'])).label;

            % Reordering
            S_slic = sp_reorder_fct(S_slic);
            S_lsc = sp_reorder_fct(S_lsc);
            S_scalp = sp_reorder_fct(S_scalp);
            S_SH = sp_reorder_fct(S_SH);
            S_SSN = sp_reorder_fct(S_SSN);
            
            % Pour ASA, moyenne sur chaque ground truth
            ASA_image_metrics_slic = [];
            ASA_image_metrics_lsc = [];
            ASA_image_metrics_scalp = [];
            ASA_image_metrics_sh = [];
            ASA_image_metrics_ssn = [];

            for gt_idx = 1:length(gt_cell)
                gt = gt_cell.groundTruth{gt_idx}.Segmentation;

                
                ASA_image_metrics_slic = [ASA_image_metrics_slic, asa_metric(S_slic, gt)];
                ASA_image_metrics_lsc = [ASA_image_metrics_lsc, asa_metric(S_lsc, gt)];
                ASA_image_metrics_scalp = [ASA_image_metrics_scalp, asa_metric(S_scalp, gt)];
                ASA_image_metrics_sh = [ASA_image_metrics_sh, asa_metric(S_SH, gt)];
                ASA_image_metrics_ssn = [ASA_image_metrics_ssn, asa_metric(S_SSN, gt)];
            end

            % Append average metrics of this image to the folder metrics
            ASA_folder_metrics_slic = [ASA_folder_metrics_slic, mean(ASA_image_metrics_slic)];
            ASA_folder_metrics_lsc = [ASA_folder_metrics_lsc, mean(ASA_image_metrics_lsc)];
            ASA_folder_metrics_scalp = [ASA_folder_metrics_scalp, mean(ASA_image_metrics_scalp)];
            ASA_folder_metrics_sh = [ASA_folder_metrics_sh, mean(ASA_image_metrics_sh)];
            ASA_folder_metrics_ssn = [ASA_folder_metrics_ssn, mean(ASA_image_metrics_ssn)];
                    
            EV_folder_metrics_slic = [EV_folder_metrics_slic, ev_metric(S_slic,I)];
            EV_folder_metrics_lsc = [EV_folder_metrics_lsc, ev_metric(S_lsc,I)];
            EV_folder_metrics_scalp = [EV_folder_metrics_scalp, ev_metric(S_scalp, I)];
            EV_folder_metrics_sh = [EV_folder_metrics_sh, ev_metric(S_SH, I)];
            EV_folder_metrics_ssn = [EV_folder_metrics_ssn, ev_metric(S_SSN, I)];
            
            GR_folder_metrics_slic = [GR_folder_metrics_slic, gr_metric(S_slic)];
            GR_folder_metrics_lsc = [GR_folder_metrics_lsc, gr_metric(S_lsc)];
            GR_folder_metrics_scalp = [GR_folder_metrics_scalp, gr_metric(S_scalp)];
            GR_folder_metrics_sh = [GR_folder_metrics_sh, gr_metric(S_SH)];
            GR_folder_metrics_ssn = [GR_folder_metrics_ssn, gr_metric(S_SSN)];
        end
        
        % Append average metrics of this folder to overall metrics
        ASA_metrics_slic = [ASA_metrics_slic, mean(ASA_folder_metrics_slic)];
        ASA_metrics_lsc = [ASA_metrics_lsc, mean(ASA_folder_metrics_lsc)];
        ASA_metrics_scalp = [ASA_metrics_scalp, mean(ASA_folder_metrics_scalp)];
        ASA_metrics_sh = [ASA_metrics_sh, mean(ASA_folder_metrics_sh)];
        ASA_metrics_ssn = [ASA_metrics_ssn, mean(ASA_folder_metrics_ssn)];
        
        EV_metrics_slic = [EV_metrics_slic, mean(EV_folder_metrics_slic)];
        EV_metrics_lsc = [EV_metrics_lsc, mean(EV_folder_metrics_lsc)];
        EV_metrics_scalp = [EV_metrics_scalp, mean(EV_folder_metrics_scalp)];
        EV_metrics_sh = [EV_metrics_sh, mean(EV_folder_metrics_sh)];
        EV_metrics_ssn = [EV_metrics_ssn, mean(EV_folder_metrics_ssn)];
        
        GR_metrics_slic = [GR_metrics_slic, mean(GR_folder_metrics_slic)];
        GR_metrics_lsc = [GR_metrics_lsc, mean(GR_folder_metrics_lsc)];
        GR_metrics_scalp = [GR_metrics_scalp, mean(GR_folder_metrics_scalp)];
        GR_metrics_sh = [GR_metrics_sh, mean(GR_folder_metrics_sh)];
        GR_metrics_ssn = [GR_metrics_ssn, mean(GR_folder_metrics_ssn)];
    end
end

% Afficher les métriques pour chaque méthode
figure,
plot(np_range, ASA_metrics_slic, 'r'); hold on;
plot(np_range, ASA_metrics_lsc, 'g'); 
plot(np_range, ASA_metrics_scalp, 'b');
plot(np_range, ASA_metrics_sh, 'c');
plot(np_range, ASA_metrics_ssn, 'm'); hold off;
title('ASA metrics for different methods');
legend('SLIC', 'LSC', 'SCALP', 'SH', 'SSN');

figure,
plot(np_range,EV_metrics_slic, 'r'); hold on;
plot(np_range,EV_metrics_lsc, 'g');
plot(np_range,EV_metrics_scalp, 'b');
plot(np_range,EV_metrics_sh, 'c'); 
plot(np_range, EV_metrics_ssn, 'm'); hold off;
title('EV metrics for different methods');
legend('SLIC', 'LSC', 'SCALP', 'SH', 'SSN');

figure,
plot(np_range,GR_metrics_slic, 'r'); hold on;
plot(np_range,GR_metrics_lsc, 'g');
plot(np_range,GR_metrics_scalp, 'b');
plot(np_range,GR_metrics_sh, 'c'); 
plot(np_range, GR_metrics_ssn, 'm'); hold off;
title('GR metrics for different methods');
legend('SLIC', 'LSC', 'SCALP', 'SH', 'SSN');

