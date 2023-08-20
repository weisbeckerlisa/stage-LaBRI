%clear all;close all;clc;


function [hh,ll] = create_isoF_figure(varargin)
% h=figure;
title('Boundary Benchmark on BSDS');
hold on;

hh=[];
ll={};
type_list = 'BSD';
if (nargin >= 1)
    type_list = varargin{1};
end
if (strcmp(type_list,'BSD')==1)
hh = plot(0.700762,0.897659,'go','MarkerFaceColor','g','MarkerEdgeColor','g','MarkerSize',10);
ll{1} = '[F = 0.787] Human';
end

p = 0.897659;
r = 0.700762;
F = 2*p.*r./(p+r);

%% isoF lines
[p,r] = meshgrid(0.01:0.01:1,0.01:0.01:1);
% [p,r] = meshgrid(0.01:0.1:1,0.01:0.1:1);
F=2*p.*r./(p+r);
[C,h] = contour3(p,r,F);

%%
% colormap green
map=zeros(256,3); map(:,1)=0.75; map(:,2)=1; map(:,3)=0.75; colormap(map);

%%
% box on;
grid on;
set(gca,'XTick',0:0.1:1);
set(gca,'YTick',0:0.1:1);
set(gca,'XGrid','on');
set(gca,'YGrid','on');
xlabel('Rappel');
ylabel('Précision');
title('');
axis square;
axis([0 1 0 1]);

end
