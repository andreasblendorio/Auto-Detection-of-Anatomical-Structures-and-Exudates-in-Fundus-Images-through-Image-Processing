%% Exudates Detection First Method Code
clc;
clear;
close all;

% Add the path of folder image
addpath('Input/Exudates');

% Load image
Image_number = 1;
RGB = imread(sprintf('A (%d).jpg',Image_number));

% Select the center of optic disc to mask it later
% figure()
% imshow(RGB)
% title('RGB Image')
% [x,y] = getpts;

% Load the center cordinates of Optic Disc
q = readmatrix('Results/Optic_Disc_Detected_with_Exudates/OpticDisc_Center_Cordinates.xlsx');
x = q(Image_number,2);
y = q(Image_number,3);

% Select the green channel
G = RGB(:,:,2); 
figure()
imshow(G)
title('Green Channel')

% Contrast-limited adaptive histogram equalization (CLAHE) is applied twice
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
title('Contrast Stretching')

% The image is complemented to change the higher-intensity features into dark pixels
G4 = imcomplement(G3);
figure()
imshow(G4)
title('Complemented Green Channel')

% Extended minima transformation
G5 = imextendedmin(G4,2);
figure()
imshow(G5)
title('Extended minima transformation')

% Opening 
radius = 5;
SE = strel('disk', radius);
G6 = imopen(G5,SE);
figure()
imshow(G6)
title('Opening operation')

% Apply circular mask to delete optic disk
circleCenterX = round(x); 
circleCenterY =  round(y); 
meanOD_Radius = 155; %155

circleImage = false(size(RGB,1), size(RGB,2)); 
[x, y] = meshgrid(1:size(RGB,2), 1:size(RGB,1)); 
circleImage((x - circleCenterX).^2 + (y - circleCenterY).^2 <= meanOD_Radius.^2) = true; 
circleImage = imcomplement(circleImage); % Color are inverted to cover inside the circle

maskedImage = bsxfun(@times, G6, cast(circleImage,class(G6)));
figure()
imshow(maskedImage)
title('Delete Optic Disk')

% Plot boundaries over the original image
figure()
imshow(RGB)
hold on 
u = contour(maskedImage,'y');
legend('Exudates Contour')
xlabel(sprintf('X axis Size: %d',size(RGB,2))) 
ylabel(sprintf('Y axis Size: %d',size(RGB,1)))
title('Fundus Image')

%%
% Load mask of truth
addpath('Verità/Exudates_Mask');
truth = imread(sprintf('A (%d).bmp',Image_number));
figure()
subplot(1,2,1)
imshow(RGB)
hold on 
contour(maskedImage,'y');
title('Exudates found')
hold off
subplot(1,2,2)
imshow(RGB)
hold on 
contour(truth,'g');
title('Truth of Exudates')
hold off

