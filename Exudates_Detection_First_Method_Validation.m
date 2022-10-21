%% Exudates Detection First Method Validation Code
clc;
clear;
close all;

% Create Folder to save results
mkdir Results/Exudates_First_Method_Detected

% Add the path of folder image
addpath('Input/Exudates');

% Compute the number of image present in the folder image
folderImage = dir(['Input/Exudates','/*.jpg']);
numImage = length(folderImage);

% Load image
for Image_number=1:numImage
RGB=imread(sprintf('A (%d).jpg',Image_number));

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

% Contrast-limited adaptive histogram equalization (CLAHE) is applied twice
G1 = adapthisteq(G);
G2 = adapthisteq(G1);

% Contrast stretching transformation
G3 = imadjust(G2, stretchlim(G2, [0.05 0.95]), []); 

% The image is complemented to change the higher-intensity features into dark pixels
G4 = imcomplement(G3);

% Extended minima transformation
G5 = imextendedmin(G4,2);

% Opening 
radius = 5;
SE = strel('disk', radius);
G6 = imopen(G5,SE);

% Apply circular mask to delete optic disk
circleCenterX = round(x); 
circleCenterY =  round(y); 
meanOD_Radius = 155;

circleImage = false(size(RGB,1), size(RGB,2)); 
[x, y] = meshgrid(1:size(RGB,2), 1:size(RGB,1)); 
circleImage((x - circleCenterX).^2 + (y - circleCenterY).^2 <= meanOD_Radius.^2) = true; 
circleImage = imcomplement(circleImage); % Color are inverted to cover inside the circle

maskedImage = bsxfun(@times, G6, cast(circleImage,class(G6)));

% Plot boundaries over the original image
fig = figure();
imshow(RGB)
hold on 
u = contour(maskedImage,'y');
legend('Exudates Contour')
xlabel(sprintf('X axis Size: %d',size(RGB,2))) 
ylabel(sprintf('Y axis Size: %d',size(RGB,1)))
title(sprintf('Fundus Image %d', Image_number))
saveas(fig,sprintf('Results/Exudates_First_Method_Detected/Fundus_Image_%d.tif', Image_number));

clear u
close all

end