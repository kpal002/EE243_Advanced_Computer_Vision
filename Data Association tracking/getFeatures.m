function feat = getFeatures(I, bbox)

% I is an image
% bbox is a N x 4 matrix, containing the x,y,w,h of each bbox and N is the number of bbox
% feat is a N x n_feat dimensional matrix where n_feat is the feature length

% YOUR CODE HERE. DO NOT CHANGE ANYTHING ABOVE THIS.

I = double(I);
feat = zeros(size(bbox,1),16);
bbox = round(bbox);
for i = 1:size(bbox,1)
    H = getHoG(I(bbox(i,2):min(bbox(i,2)+bbox(i,4),size(I,1)),bbox(i,1):min(bbox(i,1)+bbox(i,3),size(I,2))));
    feat(i,:) = transpose(H);
end