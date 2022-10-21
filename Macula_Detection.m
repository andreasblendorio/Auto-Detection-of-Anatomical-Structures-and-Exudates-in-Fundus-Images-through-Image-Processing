%% Macula Detection Code
clc;
clear;
close all;

% Add the path of folder image
addpath('Input/Non-AMD');

% Load image
Image_number = 1;
RGB = imread(sprintf('N (%d).jpg',Image_number));

% Select the center of optic disc
figure()
imshow(RGB)
title('RGB Image, select the center of Optic Disc:')
% [x,y] = getpts;   

% Load the center cordinates of Optic Disc
q = readmatrix('Results/Optic_Disc_Detected/OpticDisc_Center_Cordinates.xlsx');
x = q(Image_number,2);
y = q(Image_number,3);

% Select the green channel
G = RGB(:,:,2); 
figure()
imshow(G)
title('Green Channel')

% Contrast-limited adaptive histogram equalization (CLAHE) is applied twice.
G1 = adapthisteq(G);
G2 = adapthisteq(G1);
figure()
subplot(1,2,1)
imshow(G1)
title('First CLAHE')
subplot(1,2,2)
imshow(G2)
title('Second CLAHE')

% Contrast stretching transformation
G3 = imadjust(G2, stretchlim(G2, [0.05 0.95]), []); 
figure()
imshow(G3)
title('Contrast stretching transformation')

% Morphological closing (To delete blood vessels)
radius = 20;
SE = strel('disk', radius);
G4 = imclose(G3, SE);
figure()
imshow(G4)
title('Morphological closing')

% Formula to detect Macula: (1.5*d) < M < (3.5*d) --> (3*r) < M < (7*r)
% Mean Radius of Optic Disc and Radius Factor of the formula
meanOD_Radius = 150; 
externalRadiusFactor = 7;
internalRadiusFactor = 3;

%% Mask 1
% Initialize parameters for the external mask
circleCenterX = round(x); 
circleCenterY =  round(y); 
externalRadius = meanOD_Radius * externalRadiusFactor;

% Initialize an image to a logical image of the circle. 
circleImage = false(size(RGB,1), size(RGB,2)); 
[x, y] = meshgrid(1:size(RGB,2), 1:size(RGB,1));  % returns 2-D grid coordinates
circleImage((x - circleCenterX).^2 + (y - circleCenterY).^2 <= externalRadius.^2) = true; 

% Element-wise operation
maskedImage1 = bsxfun(@times, G4, cast(circleImage,class(G4)));

%% Mask 2
% Initialize parameters for the internal mask
internalRadius = meanOD_Radius * internalRadiusFactor; 

% Initialize an image to a logical image of the circle. 
circleImage = false(size(RGB,1), size(RGB,2)); 
[x, y] = meshgrid(1:size(RGB,2), 1:size(RGB,1));   % returns 2-D grid coordinates 
circleImage((x - circleCenterX).^2 + (y - circleCenterY).^2 <= internalRadius.^2) = true; 
circleImage = imcomplement(circleImage); % Color are inverted to cover inside the circle

% Element-wise operation
maskedImage2 = bsxfun(@times, maskedImage1, cast(circleImage,class(maskedImage1)));

%% Mask 3 over y
% The third mask on the x-axis goes from the center of the optic disc to the end of the image, 
% while on the y-axis it goes from the center +/- a height of three times the average radius of the optic disc
mask = zeros(size(RGB,1), size(RGB,2));
maculaHeight = meanOD_Radius*3;
mask(circleCenterY-maculaHeight:circleCenterY+maculaHeight, circleCenterX:size(RGB,2)) = 1;

maskedImage3 = bsxfun(@times, maskedImage2, cast(mask,class(maskedImage2)));
figure()
imshow(maskedImage3);
title('Masked Image')

%% Macula Detection
pixelValue = maskedImage3(:, :); % If it's a gray scale image.
min_val = min(maskedImage3(maskedImage3>0)); % Look for the pixel with lowest intensity> 0, to avoid selecting the background
[row,col] = find(pixelValue == min_val);

% Create a mask to search the center of Macula
mask=zeros(size(RGB,1),size(RGB,2),'logical'); 
mask(row(:),col(:)) = 255;

% Find the centroid of binarized image (mask)
centroid = regionprops(mask, 'Centroid'); 
x = centroid(1).Centroid(1); 
y = centroid(1).Centroid(2);

% Plot centroid over the original image
fig = figure(); 
imshow(RGB)
hold on
plot(x, y,'kx','MarkerSize',12,'LineWidth',2,'MarkerEdgeColor','b')
legend('Macula centroid')
xlabel(sprintf('X axis Size: %d',size(RGB,2))) 
ylabel(sprintf('Y axis Size: %d',size(RGB,1)))
title('Fundus Image')
hold off

% Center of Macula found 
C = [x y];

% Load Original cordinates of Macula
m = readmatrix('Verità/Fovea_location');
X_fovea = m(Image_number,3);
Y_fovea = m(Image_number,4);
C_fovea = [X_fovea Y_fovea];
figure()
imshow(RGB)
hold on
plot(x, y,'kx','MarkerSize',12,'LineWidth',2,'MarkerEdgeColor','b')
plot(X_fovea,Y_fovea,'kx','MarkerSize',12,'LineWidth',2,'MarkerEdgeColor','r')
legend('Macula Centroid', 'Original Macula Centroid')
title('Fundus Image')
hold off

% Calculate the offset between Original Fovea Cordinates and those found 
distance = norm(C-C_fovea);


