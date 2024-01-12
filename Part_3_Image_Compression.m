close all; clear;   % clear all plots and variables

X = imread('D:\Matlab\toolbox\images\imdata\autumn.tif');
I = rgb2gray(X);
J_1 = dct2(I);

%% diff threshold
threshold = [3, 15, 80, 200];

figure(1)
for i = 1:length(threshold)
    nz_1 = find(abs(J_1)<threshold(i));
    J_1(nz_1) = zeros(size(nz_1));
    K_1 = idct2(J_1)/255;
    subplot(2,3,i+1)
    imshow(K_1), axis off
    title(['Reconstruction Image (threshold is ', num2str(threshold(i)), ')']);
end
subplot(2,3,1)
imshow(I), axis off
title('Original Image');

%% diff block
blockSize = 32;
threshold_2 = 200;
[rows, cols] = size(I);
DCT_result = zeros(rows, cols);
for row = 1:blockSize:rows
    for col = 1:blockSize:cols
        % get current block
        current_block = I(row:min(row+blockSize-1,rows), col:min(col+blockSize-1,cols));
        dctBlock = dct2(current_block);        
        DCT_result(row:min(row+blockSize-1,rows), col:min(col+blockSize-1,cols)) = dctBlock;
    end
end
nz_2 = find(abs(DCT_result)<threshold_2);
DCT_result(nz_2) = zeros(size(nz_2));

K_2 = zeros(rows, cols);
% idct noiseless
for row = 1:blockSize:rows
    for col = 1:blockSize:cols
        current_dctBlock = DCT_result(row:min(row+blockSize-1,rows), col:min(col+blockSize-1,cols));
        idctBlock = idct2(current_dctBlock);
        K_2(row:min(row+blockSize-1,rows), col:min(col+blockSize-1,cols)) = idctBlock;
    end
end
figure(2)
imshow(K_2/255), axis off
title(['Reconstruction Image (blockSize is ', num2str(blockSize), ', threshold is ', num2str(threshold_2),')']);

%% diff threshold(PSNR)
J = dct2(I);
interval = 1;
threshold_calcu = 0:interval:200;
PSNR = zeros(size(threshold_calcu));

for i = 1:interval:length(threshold_calcu)
    nz = find(abs(J)<threshold_calcu(i));
    J(nz) = zeros(size(nz));
    K = idct2(J);
    PSNR(i) = psnr(K, double(I));
    
end

figure(3)
plot(threshold_calcu, PSNR, '-r'); 
legend('PSNR');
xlabel('Threshold') 
ylabel('PSNR') 
title('PSNR vs. Threshold')


