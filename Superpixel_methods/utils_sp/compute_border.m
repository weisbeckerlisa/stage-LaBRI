

function B = compute_border(S, I)

    [h,w,z] = size(I);
    B = zeros(h,w);
    for i=1:max(S(:))
        j = (S == i);
        bb = bwboundaries(j);
        if ~isempty(bb)
            for k=1:length(bb{1})
                B(bb{1}(k,1),bb{1}(k,2)) = 1;
            end
        end
    end
end