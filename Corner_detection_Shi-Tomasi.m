function corners = getCorners(I, ncorners)

% I is a 2D matrix 
% ncorners is the number of 'top' corners to be returned
% corners is a ncorners x 2 matrix with the 2D localtions of the corners

% FILL IN YOUR CODE HERE. DO NOT CHANGE ANYTHING ABOVE THIS.

%% part 0 - Sobel operator
    img=im2double(I);
    dx = [1 2 1; 0 0 0; -1 -2 -1]; % Sobel derivatives
    dy = transpose(dx);
    Ix = conv2(img, dx, 'same');
    Iy = conv2(img, dy, 'same');
    %Generation of Gaussian smoothing filter w(u,v) 
    w = fspecial('gaussian',5,2.5);
    %Computation of Ix2, Iy2, and Ixy
    Ix2 = conv2(Ix.^2, w, 'same');
    Iy2 = conv2(Iy.^2, w, 'same');
    IxIy = conv2(Ix.*Iy, w, 'same');
    %% part 1- Compute Matrix E which contains for every point the value
    k = 0.04;
    det = (Ix2.*Iy2) - IxIy.^2;
    trace =Ix2 + Iy2;
    R = det - k*(trace).^2;
    [x,y] = find(R>0.1);
    imshow(img);
    hold on;
    plot(y,x,'r+', 'MarkerSize', 5);
    corners = [x,y];

    