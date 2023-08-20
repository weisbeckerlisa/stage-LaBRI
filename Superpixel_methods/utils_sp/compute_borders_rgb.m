function B = compute_borders_rgb(S, I, borderColor)

    [h,w,z] = size(I);
    B = zeros(h,w,3); % Creer une image RGB
    
    for i=1:max(S(:))
        j = (S == i);
        bb = bwboundaries(j);
        if ~isempty(bb)
            for k=1:length(bb{1})
                B(bb{1}(k,1),bb{1}(k,2),:) = borderColor;
            end
        end
    end
end
