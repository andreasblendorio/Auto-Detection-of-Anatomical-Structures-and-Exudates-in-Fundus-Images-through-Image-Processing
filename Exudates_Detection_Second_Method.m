%% Exudates Detection Second Method Code
clc;
clear;
close all;

% Add the path of folder image
addpath('Input/Exudates');

% Load image
Image_number = 1;
RGB = imread(sprintf('A (%d).jpg',Image_number));

% Select the center of optic disc to mask it later
figure()
imshow(RGB)
title('RGB Image, select the center of Optic Disc:')
% [x,y] = getpts;

% Load the center cordinates of Optic Disc
q = readmatrix('Results/Optic_Disc_Detected_with_Exudates/OpticDisc_Center_Cordinates.xlsx');
x = q(Image_number,2);
y = q(Image_number,3);

% Decorrelation stretch
I_RGB_DS = decorrstretch(RGB);
figure()
imshow(I_RGB_DS)
title('Decorrelation stretch')

% Select the green channel of the decorrelated-image
I_G_DS = I_RGB_DS(:,:,2); 
figure()
imshow(I_G_DS)
title('Green Channel')

% The decorrelated-image I_RGB_DS is also converted into YCbCr color space
YCbCrImage = rgb2ycbcr(I_RGB_DS);
Y = YCbCrImage(:,:,1);
Cb = YCbCrImage(:,:,2);
Cr = YCbCrImage(:,:,3);

figure()
subplot(2, 2, 1)
imshow(YCbCrImage);
title('YCbCr Image');
subplot(2, 2, 2)
imshow(Y);
title('Y Image');
subplot(2, 2, 3)
imshow(Cb);
title('Cb Image');
subplot(2, 2, 4)
imshow(Cr);
title('Cr Image');

% By combining these two color space components a resulting image IRes is obtained:
IRes = (I_G_DS-Cb) + Y;
figure()
imshow(IRes)
title('Resulting image')

% Morphological top-hat to IRes
r1 = 20;
SE1 = strel('disk',r1);
T_hat = imtophat(IRes,SE1);
figure()
imshow(T_hat)
title('Morphological top-hat to IRes')

% Morphological bottom-hat to IRes
r2 = 200;
SE2 = strel('disk',r2);
B_hat = imbothat(IRes,SE2);
figure()
imshow(B_hat)
title('Morphological bottom-hat to IRes')

% Enhances the contrast of the image IRes based on the following formula:
I_TB = T_hat - B_hat + IRes;
figure()
imshow(I_TB)
title('Resulting image')

% Median filter
m = 9;
n = 9;
I_M = medfilt2(I_TB,[m n]);
figure()
imshow(I_M)
title('Median filter')

% Morphological top-hat to I_M
T_hat = imtophat(I_M,SE2);

% Morphological bottom-hat to I_M
B_hat = imbothat(I_M,SE1);

% Another process is performed to remove the background of the image IM based on top- and bottom-hat 
% transforms with different values of SE using the following formula:
I_F = B_hat - I_M + T_hat;
figure()
imshow(I_F)
title('Final pre-processed image IF')

% Find left threshold using Otsu's method
[counts,~] = imhist(I_F);
T = otsuthresh(counts);

% Binarization
bin = im2bw(I_F,T);  
figure()
imshow(bin)
title('Binarization')

% Apply circular mask to delete optic disk
circleCenterX = round(x); 
circleCenterY =  round(y); 
meanOD_Radius = 155; 

circleImage = false(size(RGB,1), size(RGB,2)); 
[x, y] = meshgrid(1:size(RGB,2), 1:size(RGB,1)); 
circleImage((x - circleCenterX).^2 + (y - circleCenterY).^2 <= meanOD_Radius.^2) = true; 
circleImage = imcomplement(circleImage); % Color are inverted to cover inside the circle

maskedImage = bsxfun(@times, bin, cast(circleImage,class(bin)));
figure()
imshow(maskedImage)
title('Delete Optic Disk')

% Plot boundaries over the original image
fig = figure(); 
imshow(RGB)
hold on
u = contour(maskedImage,'b');
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
contour(maskedImage,'b');
title('Exudates found')
hold off
subplot(1,2,2)
imshow(RGB)
hold on 
contour(truth,'g');
title('Truth of Exudates')
hold off