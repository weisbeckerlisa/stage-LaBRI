
function main_fct_sp(img, gt, lab_map)

img = double(img);

% Reordering
lab_map = sp_reorder_fct(lab_map);

% Adjacency
[mat_adj,borders] = sp_adjacency_fct(lab_map);

% Display
figure,
subplot 221
imagesc(uint8(img))
title('image');
pause(1);
subplot 222
imagesc(uint8(img.*borders))
title('Superpixel decomposition')
pause(1);


%% Features and evaluation

% Mean color and superpixel center
[sp_center,sp_color,sp_center_img,sp_color_img] = sp_feat_fct(lab_map,img);

% Shape regularity evaluation
[c] = c_metric(lab_map);

[gr] = gr_metric(lab_map); 
%mex -O CFLAGS="\$CFLAGS -Wall -Wextra -W -std=c99" ./utils_sp/gr_metric_mex.c -outdir ./utils_sp
%[gr] = gr_metric_mex(int32(lab_map));

% Color homogeneity evaluation
[icv] = icv_metric(lab_map,img);
[ev] = ev_metric(lab_map,img);

subplot 223
imagesc(uint8(sp_color_img.*borders));
title(sprintf('Mean colors | EV = %1.3f',ev));
subplot 224
imagesc(uint8(sp_center_img.*borders))
if (exist('gr'))
    title(sprintf('Superpixel borders | GR = %1.3f',gr));
else
    title(sprintf('Superpixel borders | C = %1.3f',c));
end
drawnow;
pause(1);

%% ASA (Achievable Segmentation Accuracy vs GT)

if (~isempty(gt)) %gt is provided
    
    [asa] = asa_metric(lab_map,gt);
    
    %mex -O CFLAGS="\$CFLAGS -Wall -Wextra -W -std=c99" ./utils_sp/asa_metric_mex.c -outdir ./utils_sp
    %[asa] = asa_metric_mex(int32(lab_map),int32(gt));
    
    figure,
    subplot 221
    imagesc(uint8(img))
    title('Image')
    subplot 222
    imagesc(uint8(img.*borders))
    title('Superpixel decomposition')
    subplot 223
    imagesc(lab_map)
    title(sprintf('Superpixel map ASA = %1.3f', asa))
    subplot 224
    imagesc(gt)
    title('Ground truth')
    pause(1);
end


%% Superpixel neighborhood display

figure,
imagesc(uint8(img.*borders))
title('Selection of a superpixel')
drawnow;
pause(1);

[x,y] = ginput(1);
lab = lab_map(round(y),round(x));
close(2)

% Neighborhood radius
R = 75;

% Selection of neighboring superpixels according to their barycenters
tmp = double(lab_map*0);
for i=1:max(lab_map(:))
    if ((sp_center(lab,1) - sp_center(i,1))^2 + (sp_center(lab,2) - sp_center(i,2))^2 < R^2)
        [tmp_i] = sp_border_fct(lab_map,i);
        tmp = tmp + tmp_i;
    end
end
tmp = repmat(tmp,[1 1 3]);
borders_lab = img;
borders_lab(lab_map == lab) = 200;
borders_lab((~borders ~= 0) & (tmp == 0)) = 0;
borders_lab(tmp ~= 0) = 255;

% Add centers
borders_lab = min(borders_lab+(255-sp_center_img),255);

figure,
imagesc(uint8(borders_lab))
title('Superpixel neighborhood display')


