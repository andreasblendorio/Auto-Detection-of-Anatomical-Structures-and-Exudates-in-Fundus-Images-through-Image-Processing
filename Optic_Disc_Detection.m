%% Optic Disc Detection Code
clc;
clear;
close all;

% Add the path of folder image
addpath('Input/Non-AMD');

% Load image
Image_number = 1;
RGB = imread(sprintf('N (%d).jpg',Image_number));
figure()
imshow(RGB)
title('RGB Image')

% Evaluate Green Channel
G = RGB(:,:,2);
figure()
imshow(G)
title('Green Channel')

% Compute mean
mean_G = mean2(G);

% Compute standard deviation
std_G = std2(G);

% Remove mean and standard deviation from Green channel
G1 = G - mean_G - std_G;
figure()
imshow(G1)
title('RGB Image')

% Size of figure
[row,col,~] = size(RGB);

% Windows heigth and width
w_factor = 14;
w_x = ceil(row/w_factor);    
w_y = ceil(col/w_factor);

% Search the window with highest mean value on the Green channel
maxMean = 0;
step_x = 23;
step_y = 29;
% Sliding window for high mean value detection
for i = 1:step_x:size(G,1)-w_x+1  
    for j = 1:step_y:size(G,2)-w_y+1
        tmp = G1(i:i+w_x-1, j:j+w_y-1, :);
        if (mean2(tmp) >= maxMean)
            maxMean = mean2(tmp);
            i_max = i;
            j_max = j;
        end
    end
end

% ROI Identification (Region of Interest)
vertex = 250;
ROI_dim = 600; 
% ROI Check borders
% X1
if j_max - vertex < 0  % Col = x
    start_x = 1;
else
    start_x = j_max - vertex;
end
% Y1
if i_max - vertex < 0   % Row = y
    start_y = 1;
else
    start_y = i_max - vertex;
end
% X2
if start_x + ROI_dim > size(RGB,2)
    end_x = size(RGB,2);
else
    end_x = start_x + ROI_dim;
end
% Y2
if start_y + ROI_dim > size(RGB,1)
    end_y = size(RGB,1);
else
    end_y = start_y + ROI_dim;
end

% Crop the RGB
[rgb_crop,rect] = imcrop(RGB,[start_x start_y end_x-start_x-1 end_y-start_y-1]);
figure()
imshow(rgb_crop)
title('ROI (Region of Interest) Optic Disk')

% Select Red Channel of ROI
R_crop = rgb_crop(:,:,1);
figure()
imshow(R_crop)
title('Red Channel ROI')

% Left crop
[r_crop_left,~] = imcrop(R_crop,[1 1 j_max-rect(1)-1 size(R_crop,1)]);  % [xmin ymin width height]
figure()
imshow(r_crop_left)
title('Left ROI Optic Disk')

% Remove columns of cropped image where there are zero values
tmp_left = r_crop_left;
tmp_left(:,any(tmp_left == 0)) = [];

% Find left threshold using Otsu's method
[counts_left,~] = imhist(tmp_left);
T_left = otsuthresh(counts_left);

% Apply binarization on left size
bin1 = im2bw(r_crop_left,T_left);  
figure()
imshow(bin1)
title('Binarization Cropped Left')

% Rigth crop
[r_crop_right,~] = imcrop(R_crop,[j_max-rect(1)+1 1 size(R_crop,1) size(R_crop,1)]);
figure()
imshow(r_crop_right)
title('Rigth ROI Optic Disk')

% Remove columns of cropped image where there are zero values
tmp_right = r_crop_right;
tmp_right(:,any(tmp_right == 0)) = [];

% Find rigth threshold using Otsu's method
[counts_right,~] = imhist(tmp_right);
T_right = otsuthresh(counts_right);

% Apply binarization on rigth size
bin2 = im2bw(r_crop_right,T_right);  
figure()
imshow(bin2)
title('Binarization Cropped Rigth')

% Concatenate binarized images
bin = cat(2,bin1,bin2); 
figure()
imshow(bin)
title('Binarized Cropped Image')

% Morphological operation (opening)
r = 5;
SE = strel('disk',r);
J = imopen(bin,SE);
figure()
imshow(J)
title('Morphological Opening')

% Take the largest blob
num_objects = 1;
binaryImage = bwareafilt(J, num_objects);
figure()
imshow(binaryImage)
title('Binarization with largest blob')

% Take convex hull (Smoothing)
J1 = bwconvhull(binaryImage, 'objects');
figure()
imshow(J1)
title('Centroid Rounding')

% Reconstruct binary image with original size from cropped binary image
mask=zeros(size(RGB,1),size(RGB,2));

nx = rect(1);   % location 
ny = rect(2);   % location 
w = size(J1,2); % width 
h = size(J1,1); % height 
% replace 
mask(ny:ny+h-1,nx:nx+w-1) = J1; 
figure()
imshow(mask)
title('Reconstruct binary image')

% Find the centroid of binarized image (mask)
centroid = regionprops(mask, 'Centroid'); 
x = centroid(1).Centroid(1); 
y = centroid(1).Centroid(2);

% Optic disk boundary 
[B,L] = bwboundaries(mask,'noholes');
boundary = cell2mat(B);

bounds(1,:) = boundary(:,2);
bounds(2,:) = boundary(:,1);

% Plot boundary and centroid over the original image
fig = figure();
imshow(RGB)
hold on
plot(bounds(1,:), bounds(2,:), 'r', 'LineWidth', 2);
plot(x, y, 'kx','MarkerSize',12,'LineWidth',2,'MarkerEdgeColor','b')
legend('Optic Disk Contour','Optic Disk Centroid')
xlabel(sprintf('X axis Size: %d',col)) 
ylabel(sprintf('Y axis Size: %d',row))
title('Fundus Image')
hold off

% Approximate calculation of the radius
C = [x y]; % Center of Optic Disc
K = zeros(1,length(bounds));
for i = 1:length(bounds)
K(i) = norm(C-bounds(:,i)'); % Saves all the point-to-point distances between the center and the single coordinate of the contour in K.
end

radius = mean2(K); % The Radius of the Optic Disc is given by the average of the distances

% Load mask of truth
addpath('Verità/OD_Mask');
truth = imread(sprintf('V (%d).bmp',Image_number));
truth = imcomplement(truth);
truth = logical(truth);
figure()
subplot(1,2,1)
imshow(mask)
title('Optic Disk Mask found')
subplot(1,2,2)
imshow(truth)
title('Truth of the Optic Disk')

% Find the centroid of mask of truth
centroid_truth = regionprops(truth, 'Centroid');
x_truth = centroid_truth(1).Centroid(1); 
y_truth = centroid_truth(1).Centroid(2);

% Original Optic disk boundary 
[B1,L1] = bwboundaries(truth,'noholes');
Original_boundary = cell2mat(B1);

Original_bounds(1,:) = Original_boundary(:,2);
Original_bounds(2,:) = Original_boundary(:,1);

% Plot boundaries over the original image
figure()
subplot(1,2,1)
imshow(RGB)
hold on
plot(bounds(1,:), bounds(2,:), 'r', 'LineWidth', 2);
plot(x, y, 'kx','MarkerSize',12,'LineWidth',2,'MarkerEdgeColor','b')
title('Optic Disk Contour found')
hold off
subplot(1,2,2)
imshow(RGB)
hold on
plot(Original_bounds(1,:), Original_bounds(2,:), 'r', 'LineWidth', 2);
plot(x_truth, y_truth, 'kx','MarkerSize',12,'LineWidth',2,'MarkerEdgeColor','b')
title('Original Optic Disk Contour')
hold off

% Calculate the offset between Original Center of Optic Disc Cordinates and those found 
C = [x y];
C_truth = [x_truth y_truth];
distance = norm(C-C_truth);

% Jaccard similarity coefficient for image segmentation 
similarity = jaccard(logical(mask),logical(truth));
similarity_percentage = similarity*100;
similarity_percentage = round(similarity_percentage,2);