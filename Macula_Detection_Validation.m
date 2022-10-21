%% Macula Detection Validation Code
clc;
clear;
close all;

center = struct([]);
correct_images = 0;

% Create Folder to save results
mkdir Results/Macula_Detected

% Add the path of folder image
addpath('Input/Non-AMD');

% Compute the number of image present in the folder image
folderImage = dir(['Input/Non-AMD','/*.jpg']);
numImage = length(folderImage);

% Load image
for Image_number=1:numImage
RGB=imread(sprintf('N (%d).jpg',Image_number));

% Select the center of optic disc
% figure()
% imshow(RGB)
% title('RGB Image, select the center of Optic Disc:')
% [x,y] = getpts;   

% Load the center cordinates of Optic Disc
q = readmatrix('Results/Optic_Disc_Detected/OpticDisc_Center_Cordinates.xlsx');
x = q(Image_number,2);
y = q(Image_number,3);

% Select the green channel
G = RGB(:,:,2); 

% Contrast-limited adaptive histogram equalization (CLAHE) is applied twice.
G1 = adapthisteq(G);
G2 = adapthisteq(G1);

% Contrast stretching transformation
G3 = imadjust(G2, stretchlim(G2, [0.05 0.95]), []); 

% Morphological closing (To delete blood vessels)
radius = 20;
SE = strel('disk', radius);
G4 = imclose(G3, SE);

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
title(sprintf('Fundus Image %d', Image_number))
hold off
saveas(fig,sprintf('Results/Macula_Detected/Fundus_Image_%d.tif', Image_number));

% Center of Macula found 
C = [x y];

% Load Original cordinates of Macula
m = readmatrix('Verità/Fovea_location');
X_fovea = m(Image_number,3);
Y_fovea = m(Image_number,4);
C_fovea = [X_fovea Y_fovea];

% Calculate the offset between Original Fovea Cordinates and those found 
distance = norm(C-C_fovea);


% Save the number of Fundus Images with center offset <= 50
if distance <= 50
    correct_images = correct_images +1;
end


% Save the number of images which have non-zero coordinates
if ((X_fovea == 0) && (Y_fovea == 0))
    numImage = numImage - 1;
end


% Save in center struct the information that will be saved in exel file
center(Image_number).Image = Image_number;
center(Image_number).X_Fovea = x;
center(Image_number).Y_Fovea = y;
center(Image_number).X_Truth_Fovea = X_fovea;
center(Image_number).Y_Truth_Fovea = Y_fovea;
center(Image_number).Center_Offset = distance;


clear correct_image
close all

end

% Calculate the Accuracy of the algorithm
accuracy = (correct_images*100)/numImage;
accuracy = round(accuracy, 2);


% Create exel file
writetable(struct2table(center), 'Results/Macula_Detected/Macula_Center_Cordinates.xlsx');
