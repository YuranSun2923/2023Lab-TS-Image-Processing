close all; clear;   % clear all plots and variables

%% Initial Setting
Image = imread('D:\Matlab\toolbox\images\imdata\cameraman.tif');
Tr_noiseless_image = imresize(Image, [128,128]);

sigma = 0.25; % Standard deviation, controlling the intensity of the noise
Tr_noisy_Image = imnoise(Tr_noiseless_image, 'gaussian', 0, sigma/255);


%% DCT
blockSize = 8;
[rows, cols] = size(Tr_noiseless_image);

% For noiseless image
DCT_noiseless_result = zeros(rows, cols);
for row = 1:blockSize:rows
    for col = 1:blockSize:cols
        % get current block
        current_block = Tr_noiseless_image(row:min(row+blockSize-1,rows), col:min(col+blockSize-1,cols));
        dctBlock = dct2(current_block);        
        DCT_noiseless_result(row:min(row+blockSize-1,rows), col:min(col+blockSize-1,cols)) = dctBlock;
    end
end

% For noisy image
DCT_noisy_result = zeros(rows, cols);
for row = 1:blockSize:rows
    for col = 1:blockSize:cols
        % get current block
        current_block = Tr_noisy_Image(row:min(row+blockSize-1,rows), col:min(col+blockSize-1,cols));
        dctBlock = dct2(current_block);        
        DCT_noisy_result(row:min(row+blockSize-1,rows), col:min(col+blockSize-1,cols)) = dctBlock;
    end
end
figure(1)
subplot(1,2,1);
imshow(DCT_noiseless_result);
subplot(1,2,2);
imshow(DCT_noisy_result);
%% Image Compression
num_noiseless = numel(DCT_noiseless_result);
num_noisy = numel(DCT_noisy_result);

test_DCT_noiseless_result = DCT_noiseless_result;
test_DCT_noisy_result = DCT_noisy_result;

Cr = 0.25; % required compression ratio

initial_threshold = 1;
interval = 1; % iteration interval
num_nz = zeros(length(1:interval:255), 1);
a = 1; % counter
% find the best threshold of noiseless case
for noiseless_threshold = initial_threshold:interval:255
    noiseless_compression = find(abs(test_DCT_noiseless_result)<noiseless_threshold);
    test_DCT_noiseless_result(noiseless_compression) = zeros(size(noiseless_compression));
    num_nz(a) = nnz(test_DCT_noiseless_result);
    a = a + 1;
end
[r, ~] = find(abs((num_nz/num_noiseless) - Cr) == min(abs((num_nz/num_noiseless) - Cr)));
noiseless_threshold = initial_threshold + (r-1)*interval;
% image compression
noiseless_compression = find(abs(DCT_noiseless_result)<noiseless_threshold);
DCT_noiseless_result(noiseless_compression) = zeros(size(noiseless_compression));

b = 1; % counter
% find the best threshold of noisy case
for noisy_threshold = initial_threshold:interval:255
    noisy_compression = find(abs(test_DCT_noisy_result)<noisy_threshold);
    test_DCT_noisy_result(noisy_compression) = zeros(size(noisy_compression));
    num_nz(b) = nnz(test_DCT_noisy_result);
    b = b + 1;
end
[r, ~] = find(abs((num_nz/num_noisy) - Cr) == min(abs((num_nz/num_noisy) - Cr)));
noisy_threshold = initial_threshold + (r-1)*interval;
% image compression
noisy_compression = find(abs(DCT_noisy_result)<noisy_threshold);
DCT_noisy_result(noisy_compression) = zeros(size(noisy_compression));


%% IDCT
noiseless_reconstruction = zeros(rows, cols);
noisy_reconstruction = zeros(rows, cols);
% idct noiseless
for row = 1:blockSize:rows
    for col = 1:blockSize:cols
        current_dctBlock = DCT_noiseless_result(row:min(row+blockSize-1,rows), col:min(col+blockSize-1,cols));
        idctBlock = idct2(current_dctBlock);
        noiseless_reconstruction(row:min(row+blockSize-1,rows), col:min(col+blockSize-1,cols)) = idctBlock;
    end
end
% idct noisy
for row = 1:blockSize:rows
    for col = 1:blockSize:cols
        current_dctBlock = DCT_noisy_result(row:min(row+blockSize-1,rows), col:min(col+blockSize-1,cols));
        idctBlock = idct2(current_dctBlock);
        noisy_reconstruction(row:min(row+blockSize-1,rows), col:min(col+blockSize-1,cols)) = idctBlock;
    end
end

%% calculate PSNR
PSNR_noisy = PSNR_calcu(double(Tr_noiseless_image), double(Tr_noisy_Image));
PSNR_noisy_reconstruction = PSNR_calcu(double(Tr_noiseless_image), double(noisy_reconstruction));
PSNR_noiseless_reconstruction = PSNR_calcu(double(Tr_noiseless_image), double(noiseless_reconstruction));


figure(2)
subplot(2,2,1);
imshow(Tr_noiseless_image, []);
title('Noiseless Image');

subplot(2,2,2);
imshow(Tr_noisy_Image, []);
title('Noisy Image');

subplot(2,2,3);
imshow(noiseless_reconstruction/255);
title('Reconstruction Image for Noiseless Case');

subplot(2,2,4);
imshow(noisy_reconstruction/255);
title('Reconstruction Image for Noisy Case');

function PSNR = PSNR_calcu(originalImage, test)
Y1=double(originalImage);
Y2=double(test);
Diff = Y1 - Y2;
MSE = sum(Diff(:).*Diff(:)) / numel(Y1);
PSNR = 10*log10(255^2 / MSE);
end

