close all

[I, map] = imread('./peppers_color.tif');
I = ind2rgb(I(:,:,1), map);

I = imresize(I, [100, 100]);
[rows, cols, c] = size(I);
N = rows * cols;
% Parameters adapted from the paper https://ieeexplore.ieee.org/document/868688
r = 2;
sig_i = 3;
sig_x = 5;
nc_threshold = 0.025;
area_threshold = 300;

V = zeros(N,c);
W = sparse(N,N); 
X_t = zeros(rows, cols, 2);
X = zeros(N,1,2);
F = zeros(N,1,c);



for k = 1:c
    cnt = 1;
    for i = 1:cols
        for j = 1:rows
            V(cnt,k) = I(j,i,k);
            F(cnt,1,k) = I(j,i,k);
            X_t(j,i,1) = j;
            X_t(j,i,2) = i;
            if k ~=3
                X(cnt,1,k) = X_t(j,i,k);
            end
            cnt = cnt + 1;        
        end
    end 
end



F = uint8(F); 

r_t = floor(r);
for m =1:cols
    for n =1:rows
        
        range_cols = (m - r_t) : (m + r_t); 
        range_rows = transpose((n - r_t) :(n + r_t));
        valid_col_index = range_cols >= 1 & range_cols <= cols;
        valid_row_index = range_rows >= 1 & range_rows <= rows;
        
        range_cols = range_cols(valid_col_index);   
        range_rows = range_rows(valid_row_index);
        
        cur_vertex = n + (m - 1) * rows;
        
        l_r = length(range_rows);
        l_c = length(range_cols);
        temp_1 = zeros(l_r,l_c);
        temp_2 = zeros(l_r,l_c);
        for i = 1:length(range_rows)
            for j = 1:length(range_cols)
                temp_1(i,j) = range_rows(i,1);
            end
        end
                   
        for i = 1:length(range_rows)
            for j = 1:length(range_cols)
                temp_2(i,j) = ((range_cols(1,j) -1) .*rows);
            end
        end
        n_vertex_temp = temp_1 + temp_2;
        n_vertex = zeros(l_r*l_c,1);
        cnt = 1;
        for i = 1:l_c
            for j = 1:l_r
                n_vertex(cnt,1) = n_vertex_temp(j,i);
                cnt = cnt + 1;        
            end
        end 
        
        X_J = zeros(length(n_vertex),1,2); 
        for k = 1:2
            for i = 1:length(n_vertex)
                X_J(i,1,k) = X(n_vertex(i,1),1, k);
            end
        end      
                
        
        X_I_temp = X(cur_vertex, 1, :);
        X_I = zeros(length(n_vertex),1,2);  
      
        for i = 1:length(n_vertex)
            for k = 1:2
                X_I(i,1,k) = X_I_temp(1,1,k);
            end
        end
        diff_X = X_I - X_J;
        diff_X = sum(diff_X .* diff_X, 3);
        
        valid_index = (sqrt(diff_X) <= r);
        n_vertex = n_vertex(valid_index);
        diff_X = diff_X(valid_index);

        F_J = zeros(length(n_vertex),1,c); 
        for i = 1:length(n_vertex)
            for k = 1:c
                a = n_vertex(i,1);
                F_J(i,1,k) = F(a,1,k);
            end
        end
        F_J = uint8(F_J);
        
        FI_temp = F(cur_vertex, 1, :);
        F_I = zeros(length(n_vertex),1,c);  
        for i = 1:length(n_vertex)
            for k = 1:c
                F_I(i,1,k) = FI_temp(1,1,k);
            end
        end
        F_I = uint8(F_I);        
        
        diff_F = F_I - F_J;
        diff_F = sum(diff_F .* diff_F, 3); 
        W(cur_vertex, n_vertex) = exp(-diff_F / (sig_i*sig_i)) .* exp(-diff_X / (sig_x*sig_x)); % for squared distance
        
    end
end

node_index = transpose(1:N); 
[node_index, Ncut] = NPartition(node_index, W, nc_threshold, area_threshold);



for i=1:length(node_index)
    seg_I_temp_1 = zeros(N, c);
    seg_I_temp_1(node_index{i}, :) = V(node_index{i}, :);
    seg_I_temp_2{i} = (reshape(seg_I_temp_1, rows, cols, c));
    seg_I{i} = uint8(seg_I_temp_2{i});
    
end

figure;
seg_length = length(seg_I);
for i=1:seg_length
    subplot(2,4,i)
    imshow(seg_I{i}*255);
    
end


function [node_index, nc_result] = NPartition(I, W, nc_th, a_th)
N = length(W);
d = sum(W, 2);
D = sparse(N,N);
for i = 1:N
    D(i,i) = d(i);
end

[Y,~] = eigs(D-W, D, 2, 'sm'); % (D - W)Y = lambda * D * Y
eig_vector_2 = Y(:, 2);

split_point = median(eig_vector_2);
split_point = fminsearch('NcutValue', split_point, [],eig_vector_2, W, D);

partition_1 = find(eig_vector_2 > split_point);
partition_2 = find(eig_vector_2 <= split_point);
Ncut_value = NcutValue(split_point, eig_vector_2, W, D);
if (length(partition_1) < a_th || length(partition_2) < a_th || Ncut_value > nc_th)
    node_index{1}   = I;
    nc_result{1} = Ncut_value; 
    return;
end

%recursive partition
[node_index_1, Ncut_1]  = NPartition(I(partition_1), W(partition_1, partition_1), nc_th, a_th);
[node_index_2, Ncut_2] = NPartition(I(partition_2), W(partition_2, partition_2), nc_th, a_th);

node_index   = cat(2, node_index_1, node_index_2);
nc_result = cat(2, Ncut_1, Ncut_2);
end
