%lab_map = regions_from_closed_contours(contour, varargin)
%
%Computes the adjacency matrix and the borders of a superpixel decomposition.
%
%
%Inputs:    - borders: Superpixel borders, or borders of regions that need
%                      to be filled.
%           - varargin: may contains 'borders_fill' option that also
%                       associate a label to pixels of the borders.
%
%Outputs:   - label_map: Each region has been filled with a different
%                        label.


function label_map = regions_from_closed_contours(borders, varargin)

borders = borders>0;

[h,w] = size(borders);
label_map = zeros(size(borders));

c = 1;
for i=2:h-1
    for j=2:w-1
       
        if ((label_map(i,j) == 0) &&  (borders(i,j) ~= 1))
            BW2 = imfill(borders, [i j], 4);
            label_map = label_map + (BW2 - borders)*c;
            c = c + 1;
        end

    end
end

if ((nargin>1) && (strcmp(varargin{1},'borders_fill')==1))
for i=1:h
    for j=1:w
        if (borders(i,j) == 1)
            win = label_map(max(i-2,1):min(i+2,h), max(j-2,1):min(j+2,w));
            win = win(win>0);
            %Median value that exists
            [~, index] = min(abs(win(:) - median(win(:))));
            label_map(i,j) = win(index);
        end
    end
end
end
end

