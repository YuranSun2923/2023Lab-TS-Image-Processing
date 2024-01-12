close all; clear;   % clear all plots and variables

%% 1
I = imread('D:\Matlab\toolbox\images\imdata\cameraman.tif');

J = histeq(I, 32);
K = histeq(I,4);

figure(1)
subplot(2,2,1), imhist(J,32);
subplot(2,2,2), imshow(J);
subplot(2,2,3), imhist(K,32);
subplot(2,2,4), imshow(K);

%% 2 
% load Template Image
Template_image = imread('D:\Matlab\toolbox\images\imdata\cameraman.tif');
Template_image = im2gray(Template_image);
hist_template = imhist(Template_image);
% Normalise to conform the law "the sum is 1" of the probability distribution
[h,w] = size(Template_image);
hist_template = hist_template/(h*w);

% load Original Image
Original_image = imresize(imread('D:\Matlab\toolbox\images\imdata\moon.tif'),[h,w]);
Original_image = im2gray(Original_image);
hist_original = imhist(Original_image);
% Normalise to conform the law "the sum is 1" of the probability distribution
[h,w] = size(Original_image);
hist_original = hist_original/(h*w);

% The histgram is given by a function
y_ori = twomodegauss(0.15, 0.05, 0.75, 0.05, 1, 0.08, 0.002);
% Normalise to conform the law "the sum is 1" of the probability distribution
y = y_ori./sum(y_ori(:));
% CDF of the Template Image
G_datafile = [];               
for i=1:256
   G_datafile = [G_datafile sum(hist_template(1:i))]; 
end
% CDF of the function
G_function = [];
for i=1:256
   G_function = [G_function sum(y(1:i))]; 
end
% CDF of the Original Image
s = [];
for i=1:256
   s = [s sum(hist_original(1:i))]; 
end

% (1)Find the index of the smallest difference between the CDF of the original image and the CDF of the provided histogram.
index_datafile = zeros(256, 1);
for i=1:256
    diff{i} = abs(G_datafile-s(i));        
    [a, index_datafile(i)] = min(diff{i});   
end
index_function = zeros(256, 1);
for i=1:256
    diff{i} = abs(G_function-s(i));        
    [a, index_function(i)] = min(diff{i});   
end
% (2) Maps from the greyscale value of the original image to the new greyscale value by index.
result_image_datafile = zeros(h,w);
for i=1:h
   for j=1:w
      result_image_datafile(i,j)=index_datafile(Original_image(i,j)+1)-1;    %由原图的灰度通过索引映射到新的灰度
   end
end
result_image_function = zeros(h,w);
for i=1:h
   for j=1:w
      result_image_function(i,j)=index_function(Original_image(i,j)+1)-1;    %由原图的灰度通过索引映射到新的灰度
   end
end

figure(2)
subplot(3, 2, 1); imshow(Original_image,[]); title('Original Image');
subplot(3, 2, 2); imhist(Original_image); title('Histogram for Original Image');


subplot(3, 2, 3); imshow(Template_image,[]); title('Template Image');
subplot(3, 2, 4); imhist(Template_image); title('Histogram for Template Image');

subplot(3, 2, 5); imshow(result_image_datafile,[]); title('Specification Image(datafile)');
subplot(3, 2, 6); imhist(uint8(result_image_datafile)); title('Histogram for Specification Image(datafile)');

figure(3)
subplot(3, 2, 1); imshow(Original_image,[]); title('Original Image');
subplot(3, 2, 2); imhist(Original_image); title('Histogram for Original Image');

x = 0:1:255; 
subplot(3, 2, 4); stem(x, y_ori, 'Marker', "none"); title('Histogram for Two Mode Gaussian function');

subplot(3, 2, 5); imshow(result_image_function,[]); title('Specification Image(function)');
subplot(3, 2, 6); imhist(uint8(result_image_function)); title('Histogram for Specification Image(function)');

%% 3 (see Slide: PART 2 Topic 3 DIP Image Filters: P38 ~ P54)
X = imread('D:\Matlab\toolbox\images\imdata\trees.tif');
I = im2gray(X);

figure(4)
% the Sobel edge detection algorithm
subplot(2,2,1), imshow(edge(I,'sobel'))
title('Result of Sobel edge detection')
% the Roberts edge detection algorithm
subplot(2,2,2), imshow(edge(I,'roberts'))
title('Result of Roberts edge detection')
% the Prewitt edge detection algorithm
subplot(2,2,3), imshow(edge(I,'prewitt'))
title('Result of Prewitt edge detection')
% the Laplacian of Gaussian (LoG) edge detection algorithm
subplot(2,2,4), imshow(edge(I,'Log'))
title('Result of "Log" edge detection')


filter_45_kernel =  [2,  1,  0; 
                     1,  0,  -1; 
                     0,  -1, -2];

filter_minus_45_kernel = [0,  1,  2; 
                          -1,  0,  1; 
                          -2,  -1, 0];

edge_45 = imfilter(double(I), filter_45_kernel) ;
edge_minus_45 = imfilter(double(I), filter_minus_45_kernel);
figure(5)
subplot(1, 3, 1); imshow(I); title('Original Image');
subplot(1, 3, 2); imshow(edge_45/255); title('Edges at +45 degrees');
subplot(1, 3, 3); imshow(edge_minus_45/255); title('Edges at -45 degrees');

%% 4 (see Slide: PART 2 Topic 3 DIP Image Filters: P24 ~ P30)
dimension = [3, 6, 9, 12]; % the neighbourhooh dimension
X = imread('autumn.tif');
I = rgb2gray(X);
J = imnoise(I,'salt & pepper');

figure(6)
for i = 1:length(dimension)
    K = medfilt2(J,[dimension(i) dimension(i)]);
    subplot(2,3,i+1), imshow(K)
    title(['Result of median filtering (neighbourhood dimension is ', num2str(dimension(i)), ')']);
end

figure(6)
subplot(2,3,1), imshow(J)
title('Image with salt & pepper noise');


function p = twomodegauss(m1, sig1, m2, sig2, A1, A2, k)
%TWOMODEGAUSS Generates a bimodal Gaussian function.
%  P = TWOMODEGAUSS(M1, SIG1, M2, SIG2, A1, A2, K)generates a bimodal,
%  Gaussian-like function in the interval [0, 1]. P  is a 256-element vector
%  normalized so that SUM(P) equals 1. The mean and standard deviation of
%  the modes are (M1, SIG1) and (M2, SIG2), respectively. A1 and A2 are the
%  amplitude values of the two modes. Since the output is normalize, only
%  the relative values of A1 and A2 are important. K is an offset value
%  that raises the “floor” of the function. A good set of values
%  to try is M1 = 0.15, SIG1 = 0.05, M2 = 0.75, SIG2 = 0.05, A1 = 1, A2 =
%  0.07,and K = 0.002.
 
c1 = A1 * (1 / ((2 * pi) ^ 0.5) * sig1);
k1 = 2 * (sig1 ^ 2);
c2 = A2 * (1 / ((2 * pi) ^ 0.5) * sig2);
k2 = 2 * (sig2 ^ 2);
z = linspace(0, 1, 256);

p = k+c1*exp(-((z-m1).^2)./k1)+c2*exp(-((z-m2).^2)./k2);

end