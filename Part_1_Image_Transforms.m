close all; clear;   % clear all plots and variables

%% Part_1

%% 1 FFT
Standard_images = imread ('D:\Matlab\toolbox\images\imdata\cameraman.tif');
%%%%%%%%%% (a) FFT %%%%%%%%%%%
fft_Image = fft2(double(Standard_images));
% find log magnitude of the FFT
% The fftshift function is used to shift the zero frequency component of the FFT to the centre of the image

c = 255/log(1 + max(fft_Image(:)));
log_fft_Standard_images = c * log(1 + abs(fftshift(fft_Image)));

figure(1)
subplot(1, 2, 1);
imshow(Standard_images,[]);
title('Original Image');

subplot(1, 2, 2);
imagesc(log_fft_Standard_images);
daspect([1 1 1]);
colormap(subplot(1, 2, 2), 'jet'); 
colorbar; 
title('Display of the Logarithmic Amplitude of FFT');


%%%%%%%%%% (b) %%%%%%%%%%
% Create an artificial image of a grid of impulses
Grid_impulse = zeros(256, 256);
Grid_impulse(1:32:end, 1:32:end) = 255; % Place impulses at regular intervals
% fft of the Grid
fft_Grid = fft2(double(Grid_impulse));
% find log magnitude of the FFT
c = 255/log(1 + max(fftshift(fft_Grid(:))));
log_fft_Grid = c * log(1 + abs(fftshift(fft_Grid)));

figure(2)
subplot(1,2,1);
imshow(Grid_impulse,[]);
title('Original Grid');

subplot(1, 2, 2);
imagesc(log_fft_Grid);
daspect([1 1 1]);
colormap(subplot(1, 2, 2), 'gray'); 
colorbar; 
title('Display of the Logarithmic Amplitude of FFT (impulses)');

%  Regular periodic signal(sin)
% gen blank grid 
[x, y] = meshgrid(1:256, 1:256);
freq_x = 1/32;
freq_y = 1/32;
% gen sin witn 128 mean
sin_signal = 128 + 127 * sin(2 * pi * freq_x * x + 2 * pi * freq_y * y);
% fft of the sin
fft_Sin = fft2(double(sin_signal));
% find log magnitude of the FFT
c = 255/log(1 + max(fft_Sin(:)));
log_fft_Sin = c * log(1 + abs(fftshift(fft_Sin)));

figure(3)
subplot(1,2,1);
imshow(sin_signal,[]);
title('Original sin signal');

subplot(1, 2, 2);
imagesc(log_fft_Sin);
daspect([1 1 1]);
colormap(subplot(1, 2, 2), 'gray'); 
colorbar; 
title('Display of the Logarithmic Amplitude of FFT (Regular periodic signal "sin")');

% nonperiodic signals(noise)
Noise = randn(256, 256);
% fft of the noise
fft_Noise = fft2(double(Noise));
% find log magnitude of the FFT
c = 255/log(1 + max(fft_Noise(:)));
log_fft_Noise = log(1 + abs(fftshift(fft_Noise)));

figure(4)
subplot(1,2,1);
imshow(Noise,[]);
title('Original noise');

subplot(1, 2, 2);
imagesc(log_fft_Noise);
daspect([1 1 1]);
colormap(subplot(1, 2, 2), 'jet'); 
colorbar; 
title('Display of the Logarithmic Amplitude of FFT (nonperiodic signals "noise")');

%%%%%%%%%% (c) %%%%%%%%%%
% load images A and B
Image_A = imread('obj1.png');
Image_B = imread('obj2.png');

Image_A = rgb2gray(Image_A);
Image_B = rgb2gray(Image_B);
% phase of A
fft_A = fft2(double(Image_A));
phase_A = angle(fft_A);
% Amplitude of B
fft_B = fft2(double(Image_B));
amplitude_B = abs(fft_B);

% Combined fft results
Comblined_fft = amplitude_B .* exp(1j * phase_A);

% ifft of Combined fft
Reconstruct_Image = ifft2(Comblined_fft);

figure(5)
subplot(1,3,1);
imshow(Image_A,[]);
title('Image A')

subplot(1,3,2);
imshow(Image_B,[]);
title('Image B')

subplot(1,3,3);
imshow(Reconstruct_Image,[]);
title('Reconstruct Image of Combined fft')

%% 2 DCT
%%%%%%%%%% (a) %%%%%%%%%%%
Original_images = imread ('D:\Matlab\toolbox\images\imdata\autumn.tif');
Gray_image = rgb2gray(Original_images);
DCT_autumn = dct2(double(Gray_image));
FFT_autumn = fft2(double(Gray_image));

figure(6)
colormap(jet(64)), imagesc(log(abs(DCT_autumn))), colorbar
title('Log Magnitude of DCT(colormap)')

figure(7)
colormap(jet(64)), imagesc(log(abs(FFT_autumn))), colorbar
title('Log Magnitude of FFT(colormap)')

%% 3 Hadamard
image = imread ('D:\Matlab\toolbox\images\imdata\cameraman.tif');
[Hadmard_nonordered_result, H_nonordered, Hadmard_ordered_result, H_ordered] = hadamard_T(image);



figure(8)
subplot(1,3,1);
imshow(image,[]);
title('Original image');

subplot(1, 3, 2);
imagesc(log(abs(abs(Hadmard_nonordered_result))));
daspect([1 1 1]);
colormap(subplot(1, 3, 2), "gray")
colorbar
title('Log Magnitude of non-ordered Hadamard Transform');

subplot(1, 3, 3);
imshow(H_nonordered, []);
title('non-ordered Hadamard matrix');

figure(9)
subplot(1,3,1);
imshow(image,[]);
title('Original image');

subplot(1, 3, 2);
imagesc(log(abs(abs(Hadmard_ordered_result))));
daspect([1 1 1]);
colormap(subplot(1, 3, 2), "gray")
colorbar
title('Log Magnitude of ordered Hadamard Transform');

subplot(1, 3, 3);
imshow(H_ordered, []);
title('ordered Hadamard matrix');

function [Hadmard_nonordered_result, H_nonordered, Hadmard_ordered_result, H_ordered] = hadamard_T(photo)
    % Get the size of the input matrix
    [N, M] = size(photo);
    
    % Initialize Hadamard matrix
    H_nonordered = zeros(N, M);
    if log2(N) ~= round(log2(N))
        error('Size must be a power of 2 for Hadamard matrix.');
    end
    n = log2(N);
    % gen non-ordered Hadamard matrix
    H_nonordered = 1;
    for a = 1:n
    H_nonordered = [H_nonordered, H_nonordered; H_nonordered, -H_nonordered];
    end
    % gen ordered Hadamard matrix
    H_ordered = fwht(H_nonordered(:,1));
    for i = 2:N
        H_ordered = order_hadamard(H_nonordered);
    end

    Hadmard_nonordered_result = H_nonordered * double(photo)*H_nonordered;
    Hadmard_ordered_result = H_ordered * double(photo) * H_ordered;
end

function H_ordered = order_hadamard(H_unordered)
    % Get the size of the matrix
    N = size(H_unordered, 1);

    sequency = zeros(N, 1);
    % Compute the sequency (number of zero crossings) for each row
    for i = 1:N
        sequency(i) = sum(abs(diff(H_unordered(i, :))) == 2);
    end

    % Sort the rows by sequency
    [~, idx] = sort(sequency);
    H_ordered = H_unordered(idx, :);
end

