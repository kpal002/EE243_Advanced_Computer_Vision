function bbox = getDetections(D)

% D is a sum of difference image
% bbox is a N x 4 matrix, containing the x,y,w,h of each bbox and N is the number of bbox

% YOUR CODE HERE. DO NOT CHANGE ANYTHING ABOVE THIS.

d = D./max(D(:,:));
    d = imgaussfilt(medfilt2(d, [5, 5]), 0.5) > 0.5;

    se_d = strel('disk', 20);
    c = imclose(d, se_d);
    se_l = strel('line', 2, 10);
    c = imopen(c, se_l);
    CC = bwconncomp(c);
    stats = regionprops(CC, 'BoundingBox', 'Area');
    props = struct2cell(stats);
    props = transpose(props);
    props = flipud(sortrows(props, 1));
    N = size(props,1);    
    cnt = 0;
    for i = 1:N
        cur_box = props{i,2};
        bbox(i,:) = cur_box;
        cnt=cnt+1;
        if cnt >= 5
            break
        end
    end
    if N == 0 
        bbox = zeros(0,4);
    end
end