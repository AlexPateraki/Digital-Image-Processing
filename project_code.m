close all
clear 
clc
%Pateraki Alexandra 2016030154
% question_1
%https://www.mathworks.com/matlabcentral/fileexchange/53189-bit-plane-slicing
folder = 'C:\Users\Administrator\Desktop';
% Read image
A = imread(fullfile(folder,'Q16.tif'));
% Read image size
[m,n] = size(A);
% convert the image class from "uint8" to double
b = double(A);
% convert each pixel into binary using matlab command "de2bi"
c = de2bi(b);
% calling the LSB Bit of each pixel 
c1 = c(:,1);
r1 = reshape(c1,256,256);
% similarly calling every bit and converting into an arry of size 256X256
% 2nd Bit plane
c2 = c(:,2);
r2 = reshape(c2,256,256);
% 3rd Bit Plane
c3 = c(:,3);
r3 = reshape(c3,256,256);
% 4th Bit Plane
c4 = c(:,4);
r4 = reshape(c4,256,256);
% 5th Bit Plane
c5 = c(:,5);
r5 = reshape(c5,256,256);
% 6th Bit Plane
c6 = c(:,6);
r6 = reshape(c6,256,256);
% 7th Bit Plane
c7 = c(:,7);
r7 = reshape(c7,256,256);
% 8th Bit Plane
c8 = c(:,8);
r8 = reshape(c8,256,256);
% Displaying all the Bit Planes
subplot(241);
imshow(r1);title('LSB Bit Plane');
subplot(242);
imshow(r2);title('2nd Bit Plane');
subplot(243);
imshow(r3);title('3rd Bit Plane');
subplot(244);
imshow(r4);title('4th Bit Plane');
subplot(245);
imshow(r5);title('5th Bit Plane');
subplot(246);
imshow(r6);title('6th Bit Plane');
subplot(247);
imshow(r7);title('7th Bit Plane');
subplot(248);
imshow(r8);title('MSB Bit Plane');
suptitle('Question 1-bit planes'); 

Image = uint8(r1 + 2*r2 + 4*r3  + 8*r4 + 16*r5 + 32*r6 + 64*r7 + 128*r8);
figure;imshow(Image);title('reconstructed Image');
suptitle('Question 1-reconstruction the image'); 



%question_6
%https://www.mathworks.com/matlabcentral/fileexchange/25302-image-halftoning-by-floyd-s-method
%show clearly the path
folder = 'C:\Users\Administrator\Desktop';
% Read image
A = imread(fullfile(folder, 'Q14.tif')); 
halftone_img = floydHalftone(A);
figure;
subplot(1,2,1);imshow(A); title('Original');
subplot(1,2,2);imshow(halftone_img);title('Floyd Halftoned Image');
suptitle('Question 6'); 

%question_7
%reference: https://www.imageeprocessing.com/2011/06/local-histogram-equalization.html
A=imread(fullfile(folder,'Q14.tif'));
Img=A;

%WINDOW SIZE
Mgiven = 'Question 7: The size of the window is [mxn]. Insert m ';
M=input(Mgiven);
Ngiven = 'Question 7: Insert n ';
N=input(Ngiven);
mid_val=round((M*N)/2);

%FIND THE NUMBER OF ROWS AND COLUMNS TO BE PADDED WITH ZERO
in=0;
for i=1:M
    for j=1:N
        in=in+1;
        if(in==mid_val)
            PadM=i-1;
            PadN=j-1;
            break;
        end
    end
end
%PADDING THE IMAGE WITH ZERO ON ALL SIDES
B=padarray(A,[PadM,PadN]);

for i= 1:size(B,1)-((PadM*2)+1)

    for j=1:size(B,2)-((PadN*2)+1)
        cdf=zeros(256,1);
        inc=1;
        for x=1:M
            for y=1:N
  %FIND THE MIDDLE ELEMENT IN THE WINDOW          
                if(inc==mid_val)
                    ele=B(i+x-1,j+y-1)+1;
                end
                    pos=B(i+x-1,j+y-1)+1;
                    cdf(pos)=cdf(pos)+1;
                   inc=inc+1;
            end
        end

        %COMPUTE THE CDF FOR THE VALUES IN THE WINDOW
        for l=2:256
            cdf(l)=cdf(l)+cdf(l-1);
        end
            Img(i,j)=round(cdf(ele)/(M*N)*255);
     end
end
figure;
subplot(1,2,1);imshow(A);title('Before Local Histogram Equalization'); 
subplot(1,2,2);imshow(Img); title('After Local Histogram Equalization'); 
suptitle('Question 7'); 

% Question_9
%ref
%https://www.mathworks.com/matlabcentral/answers/459170-image-compression-huffman-coding
folder = 'C:\Users\Administrator\Desktop';

%i
%Reading image-convert to gray-meisure the size of the image
a= imread(fullfile(folder,'Q11.tif'));
figure,subplot(1,2,1),imshow(a), title('original image');
I = gray2ind(a,256);
subplot(1,2,2),imshow(I), title('grayscaled')
[m,n]=size(I);
Totalcount=m*n;

%ii
%variables using to find the probability
cnt=1;
sigma=0;

%iii
%computing the cumulative probability.
for i=0:255
k=I==i;
count(cnt)=sum(k(:));
%pro array is having the probabilities
pro(cnt)=count(cnt)/Totalcount;
sigma=sigma+pro(cnt);
cumpro(cnt)=sigma;
cnt=cnt+1;
end
%Symbols for an image
symbols = 0:255;
%Huffman code Dictionary
dict = huffmandict(symbols,pro);
%function which converts array to vector
vec_size = 1;
for p = 1:m
for q = 1:n
newvec(vec_size) = I(p,q);
vec_size = vec_size+1;
end
end
%Huffman Encodig
hcode = huffmanenco(newvec,dict);
%Huffman Decoding
dhsig1 = huffmandeco(hcode,dict);
%convertign dhsig1 double to dhsig uint8
dhsig = uint8(dhsig1);
%vector to array conversion
dec_row=sqrt(length(dhsig));
dec_col=dec_row;
%variables using to convert vector 2 array
arr_row = 1;
arr_col = 1;
vec_si = 1;
for x = 1:m
for y = 1:n
back(x,y)=dhsig(vec_si);
arr_col = arr_col+1;
vec_si = vec_si + 1;
end
arr_row = arr_row+1;
end
%converting image from grayscale to rgb
[deco, map] = gray2ind(back,256);
RGB = ind2rgb(deco,map);
 
imwrite(RGB,fullfile(folder,'decoded.jpeg'));

disp('In Question 9 the sizes of the images are:');
before=imfinfo(fullfile(folder,'Q11.tif'));
size=before.FileSize;
str1=['The size before was ' num2str(size) ' kB'];
disp(str1);

after=imfinfo(fullfile(folder,'decoded.jpeg'));
sizeAfter=after.FileSize;
str2=['The size after compression is ' num2str(sizeAfter) ' kB'];
disp(str2);
clear
%predictive coding
%https://www.imageeprocessing.com/2014/02/lossless-predictive-coding.html
folder = 'C:\Users\Administrator\Desktop'; 
imageNew= imread(fullfile(folder,'Q11.tif'));
%Read the input image
 figure,subplot(1,2,1),imshow(imageNew),title('Before predictive encoding');
imageNew=double(imageNew);
e=imageNew;
%Perform prediction error
for i = 1:size(imageNew,1)
    for j = 2:size(imageNew,2)
        e(i,j)=e(i,j)-imageNew(i,j-1);
    end
end

%Huffman coding
C=reshape(e,[],1);
[D1,x]=hist(C,min(min(e)):max(max(e)));
sym=x(D1>0);
prob=D1(D1>0)/numel(e);
[dict,avglen] = huffmandict(sym,prob);
comp = huffmanenco(C,dict);

%Huffman Decoding
dsig = huffmandeco(comp,dict);
e=reshape(dsig,size(imageNew,1),size(imageNew,2));
d=e;

for i = 1:size(imageNew,1)
    for j = 2:size(imageNew,2)
        d(i,j)=d(i,j-1)+e(i,j);
    end
end

%Decompressed Image
subplot(1,2,2),imshow(uint8(d)),title('After predictive encoding');
imwrite(uint8(d),fullfile(folder,'predictive encoding.jpeg'));
suptitle('Question 9-predictive encoding'); 
after2=imfinfo(fullfile(folder,'predictive encoding.jpeg'));
sizeAfter2=after2.FileSize;
str3=['The size after predictive encoding compression is ' num2str(sizeAfter2) ' kB'];
disp(str3);

%question_11
folder = 'C:\Users\Administrator\Desktop';
% Read image
img = imread(fullfile(folder, 'Q11.tif')); 
figure;
subplot(1,2,1)
imshow(img);
 title('before reducing level intensity')
intensity='Question 11: Insert the wanted level of intensity ';
in=input(intensity);
 for c=1:in
    img = uint8(img / 2);     
 end
 subplot(1,2,2)
 imshow(img);
 title(['reduce level intensity for ' num2str(in) ' time(s)'])
 suptitle('Question 11')
 
%question_14
%reference:
%https://www.mathworks.com/matlabcentral/answers/405295-custom-image-spatial-filtering-code-using-loops-not-giving-the-same-result-as-using-the-built-in-imt
%show clearly the path
folder = 'C:\Users\Administrator\Desktop';
% Read image
A = imread(fullfile(folder, 'Q11.tif')); 
% Read Kernels
prompt = 'Question 14:Select value for coefficient ';
Kernel_1 = (input(prompt))*ones(3);
prompt2 = 'Question 14: Select value for coefficient ';
Kernel_2 = (input(prompt))*ones(7);
 % Set current kernel
Kernel = Kernel_2;         
img_out = conv2(A,Kernel);   %Perform convolution on image and selected kernel
img_out_filter = imfilter(A,Kernel,'same','conv');
%%Display output images
figure('color','w')
subplot(1,2,1); imshow(A,[]); title('Original') 
subplot(1,2,2); imshow(img_out_filter,[]); title('Spatial filtering')
suptitle('Question 14-spatial filtering'); 

figure;
%laplacian enhancement
%reference https://www.mathworks.com/matlabcentral/answers/107507-laplacian-and-sobel-for-image-processing
% Compute Laplacian
laplacianKernel = [-1,-1,-1;-1,8,-1;-1,-1,-1]/8;
laplacianImage = imfilter(double(A), laplacianKernel);
% Display the image.
subplot(1,2,1); imshow(A,[]); title('Original') 
subplot(1, 2, 2);
imshow(laplacianImage);
title('Laplacian Image');
suptitle('Question 14-laplacian enhancement'); 


function outImg  = floydHalftone(inImg)
inImg = double(inImg);

[M,N] = size(inImg);
T = 127.5; %Threshold
y = inImg;
error = 0;

for rows = 1:M-1  
    
    %Left Boundary of Image
    outImg(rows,1) =255*(y(rows,1)>=T);
    error = -outImg(rows,1) + y(rows,1);
    y(rows,1+1) = 7/16 * error + y(rows,1+1);
    y(rows+1,1+1) = 1/16 * error + y(rows+1,1+1);
    y(rows+1,1) = 5/16 * error + y(rows+1,1);
    
    for cols = 2:N-1
        %Center of Image
        outImg(rows,cols) =255*(y(rows,cols)>=T);
        error = -outImg(rows,cols) + y(rows,cols);
        y(rows,cols+1) = 7/16 * error + y(rows,cols+1);
        y(rows+1,cols+1) = 1/16 * error + y(rows+1,cols+1);
        y(rows+1,cols) = 5/16 * error + y(rows+1,cols);
        y(rows+1,cols-1) = 3/16 * error + y(rows+1,cols-1);
    end
    
    %Right Boundary of Image
    outImg(rows,N) =255*(y(rows,N)>=T);
    error = -outImg(rows,N) + y(rows,N);
    y(rows+1,N) = 5/16 * error + y(rows+1,N);
    y(rows+1,N-1) = 3/16 * error + y(rows+1,N-1);
    
end

%Bottom & Left Boundary of Image
rows = M;
outImg(rows,1) =255*(y(rows,1)>=T);
error = -outImg(rows,1) + y(rows,1);
y(rows,1+1) = 7/16 * error + y(rows,1+1);

%Bottom & Center of Image
for cols = 2:N-1
    outImg(rows,cols) =255*(y(rows,cols)>=T);
    error = -outImg(rows,cols) + y(rows,cols);
    y(rows,cols+1) = 7/16 * error + y(rows,cols+1);
end

%Thresholding
outImg(rows,N) =255*(y(rows,N)>=T);

outImg = imbinarize(uint8(outImg));
end

