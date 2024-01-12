close all; clear;   % clear all plots and variables

original_image = imread('D:\Matlab\toolbox\images\imdata\cameraman.tif');

% generate noise
SNR_dB = 20;
% noise = sqrt(var(im2double(original_image(:))) / (10^(SNR_dB/10))) * randn(size(original_image));
% Generate Gaussian degradation kernel
kernelSize = 5;
h = fspecial('gaussian', [kernelSize, kernelSize], 1);
% Padding
% radius_image = (kernelSize - 1)/2;
% radius_kernel = (length(original_image) + 2)/2;
% padded_image = padarray(double(original_image), [radius_image, radius_image], 0);
% padded_h = padarray(double(h), [radius_kernel, radius_kernel], 0');

% degradation
degradation_image = imfilter(im2double(original_image), h, "conv");
y = awgn(degradation_image, SNR_dB);
noise = y - degradation_image;
% y = degradation_image + noise;

% Inverse filtering
Inverse_filtering_result = inverseFilter(y,h);
% Wiener Filtering
WienerFiltering_result = WienerFiltering(y,h,noise);

% PSNR_Degradation = PSNR_calcu(double(original_image), double(y));
% PSNR_Inverse_filtering = PSNR_calcu(double(original_image), double(Inverse_filtering_result));
% PSNR_WienerFiltering = PSNR_calcu(double(original_image), double(WienerFiltering_result));

figure(1);
subplot(2,2,1);
imshow(original_image,[]);
title('Original Image');

subplot(2,2,2);
imshow(im2uint8(y));
title('Degradation image');

subplot(2,2,3);
imshow(im2uint8(Inverse_filtering_result),[]);
title('Result of Inverse filtering');

subplot(2,2,4);
imshow(im2uint8(WienerFiltering_result),[]);
title('Result of Wiener Filtering');

% function PSNR = PSNR_calcu(originalImage, test)
% Y1=double(originalImage);
% Y2=double(test);
% Diff = Y1 - Y2;
% MSE = sum(Diff(:).*Diff(:)) / numel(Y1);
% PSNR = 10*log10(255^2 / MSE);
% end

%% Inverse filtering
function restored_image = inverseFilter(y,h)
N = size(y,1);  
fft_Y = fft2(y);   
fft_H = fft2(h,N,N); 

F = (fft_H' .* fft_Y)./(fft_H .* fft_H');
restored_image = real(ifft2(F));   
end

%% Wiener filtering
function restored_image = WienerFiltering(y,h,noise)
N = size(y,1);  
fft_Y = fft2(y);   
fft_H = fft2(h,N,N); 
fft_n = fft2(noise);

% get the psd of y
% In practical cases where a single copy of the degraded image is available, 
% it is quite common to use 𝑆𝑦𝑦(𝑢, 𝑣) as an estimate of 𝑆𝑓𝑓(𝑢, 𝑣)
S_yy = abs(fft_Y).^2 ./ numel(fft_Y);
% get the psd of noise
S_nn = abs(fft_n).^2 ./ numel(fft_n);

F = ((S_yy .* fft_H') ./ (S_yy .* (fft_H .* fft_H') + S_nn)) .* fft_Y;
restored_image = real(ifft2(F));   
end











