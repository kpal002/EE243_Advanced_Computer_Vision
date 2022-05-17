function sod = getSumOfDiff(I)

% I is a 3D tensor of image sequence where the 3rd dimention represents the time axis

% YOUR CODE HERE. DO NOT CHANGE ANYTHING ABOVE THIS.
        [m, n, N] = size(I);
    sod = zeros(m, n);
    for i = 1:N-1
        for j = i:N
            sod(:,:) = abs(I(:,:,i)-I(:,:,j))/(N*(N-1)/2);
        end
    end
    
end