img1 = im2double(imread('frame1.jpg'));
img2 = im2double(imread('frame2.jpg'));
img1=imresize(img1,[288 352])
img2=imresize(img2,[288 352])
%% 
hbm = vision.BlockMatcher('ReferenceFrameSource',...
        'Input port','BlockSize',[15 15],'SearchMethod','Exhaustive');
hbm.OutputValue = 'Horizontal and vertical components in complex form';
halphablend = vision.AlphaBlender;
%% 
motion = hbm(img1,img2);
img12 = halphablend(img2,img1);
%% 
[X,Y] = meshgrid(1:15:size(img1,2),1:15:size(img1,1));         
imshow(img12)
hold on
quiver(X(:),Y(:),real(motion(:)),imag(motion(:)),0)
hold off
%% pixel pixel
% optical flow
optic_hs = opticalFlowHS
hs_flow = estimateFlow(optic_hs,img1);
hs_flow = estimateFlow(optic_hs,img2);
%% 
optic_lk=opticalFlowLK
lk_flow = estimateFlow(optic_lk,img1);
lk_flow = estimateFlow(optic_lk,img2);
%% 
optic_farneback=opticalFlowFarneback
fb_flow=estimateFlow(optic_farneback,img1);
fb_flow=estimateFlow(optic_farneback,img2);
%% 
optic_dog=opticalFlowLKDoG
dog_flow=estimateFlow(optic_dog,img1);
dog_flow=estimateFlow(optic_dog,img2);
%% 
lk_predicted=optComp(img1,lk_flow)
hs_predicted=optComp(img1,hs_flow)
fb_predicted=optComp(img1,fb_flow)
dog_predicted=optComp(img1,dog_flow)

lk_psnr=imgPSNR(img2, lk_predicted, 255);
hs_psnr=imgPSNR(img2, hs_predicted, 255);
fb_psnr=imgPSNR(img2, fb_predicted, 255);
dog_psnr=imgPSNR(img2, dog_predicted, 255);

%% 
h = figure;
movegui(h);
hViewPanel = uipanel(h,'Position',[0 0 1 1],'Title','Plot of Optical Flow Vectors');
hPlot = axes(hViewPanel);
imshow(img12)
hold on
plot(flow,'DecimationFactor',[5 5],'ScaleFactor',60,'Parent',hPlot);
hold off
%% 
p=7
mbSize=8
%exhaustive search
[motionVect, computations_es] = motionEstES(img2,img1,mbSize,p);
imgComp = motionComp(img1, motionVect, mbSize);
ESpsnr = imgPSNR(img2, imgComp, 255);
EScomputations = computations_es;
%% 
%three step search
[motionVect_tss, computations_tss] = motionEstTSS(img2,img1,mbSize,p);
imgComp2 = motionComp(img1, motionVect_tss, mbSize);
TSSpsnr = imgPSNR(img2, imgComp2, 255);
TSScomputations = computations_tss;
%%
function [predicted2]= optComp(img1,flow)
    [y,x]=size(img1);
    predicted2=zeros(y,x);
    for j = 1:y
        for i=1:x
            dy=j+flow.Vy(j,i);
            dx=i+flow.Vx(j,i);
            %round
            dy=ceil(dy);
            dx=ceil(dx);
            predicted2(dy,dx)=img1(j,i);
        end
        disp(j)
    end
    for j=1:y
        for i=1:x
            if predicted2(j,i)== 0
                predicted2(j,i)=img1(j,i);
            end
        end
    end
end

%%
function cost = costFuncMAD(block,refBlock, n)
error = 0;
for i = 1:n
    for j = 1:n
        error = error + abs((block(i,j) - refBlock(i,j)));
    end
end
cost = error / (n*n);
end

function [dx, dy, min] = minCost(costs)
[row, col] = size(costs);
min = 65537;
for i = 1:row
    for j = 1:col
        if (costs(i,j) < min)
            min = costs(i,j);
            dx = j; dy = i;
        end
    end
end
end


function [motionVect, exhaustiveComputations] = motionEstES(imgP, imgI, macroBlockSize, p)
[row col] = size(imgI);
vectors = zeros(2,row*col/macroBlockSize^2);
costs = ones(2*p + 1, 2*p +1) * 65537;
computations = 0;
macroBlockCount = 1;
for i = 1 : macroBlockSize : row-macroBlockSize+1
    for j = 1 : macroBlockSize : col-macroBlockSize+1
        for m = -p : p        
            for n = -p : p
                refBlkVer = i + m;   
                refBlkHor = j + n;   
                if ( refBlkVer < 1 || refBlkVer+macroBlockSize-1 > row ...
                        || refBlkHor < 1 || refBlkHor+macroBlockSize-1 > col)
                    continue;
                end
                costs(m+p+1,n+p+1) = costFuncMAD(imgP(i:i+macroBlockSize-1,j:j+macroBlockSize-1), ...
                     imgI(refBlkVer:refBlkVer+macroBlockSize-1, refBlkHor:refBlkHor+macroBlockSize-1), macroBlockSize);
                computations = computations + 1;
                
            end
        end
        
        [dx, dy, min] = minCost(costs); 
        vectors(1,macroBlockCount) = dy-p-1;  
        vectors(2,macroBlockCount) = dx-p-1; 
        macroBlockCount = macroBlockCount + 1;
        costs = ones(2*p + 1, 2*p +1) * 65537;
    end
end
motionVect = vectors;
exhaustiveComputations = computations/(macroBlockCount - 1);
end


function imgComp = motionComp(imgI, motionVector, macroBlockSize)
[row col] = size(imgI);
macroBlockCount = 1;
for i = 1:macroBlockSize:row-macroBlockSize+1
    for j = 1:macroBlockSize:col-macroBlockSize+1
        dy = motionVector(1,macroBlockCount);
        dx = motionVector(2,macroBlockCount);
        refBlkVer = i + dy;
        refBlkHor = j + dx;
        imageComp(i:i+macroBlockSize-1,j:j+macroBlockSize-1) = imgI(refBlkVer:refBlkVer+macroBlockSize-1, refBlkHor:refBlkHor+macroBlockSize-1);
    
        macroBlockCount = macroBlockCount + 1;
    end
end
imgComp = imageComp;
end


function psnr = imgPSNR(imgP, imgComp, n)
[row col] = size(imgP);
err = 0;
for i = 1:row
    for j = 1:col
        err = err + (imgP(i,j) - imgComp(i,j))^2;
    end
end
mse = err / (row*col);
psnr = 10*log10(n*n/mse);
end


function [motionVector, TSScomputations] = motionEstTSS(imgP, imgI, macroBlockSize, p)
[row col] = size(imgI);
vectors = zeros(2,row*col/macroBlockSize^2);
costs = ones(3, 3) * 65537;
computations = 0;
L = floor(log10(p+1)/log10(2));   
stepMax = 2^(L-1);
macroBlockCount = 1;
for i = 1 : macroBlockSize : row-macroBlockSize+1
    for j = 1 : macroBlockSize : col-macroBlockSize+1
        costs(2,2) = costFuncMAD(imgP(i:i+macroBlockSize-1,j:j+macroBlockSize-1), ...
                                    imgI(i:i+macroBlockSize-1,j:j+macroBlockSize-1),macroBlockSize);
        x = j;
        y = i;
        computations = computations + 1;
        stepSize = stepMax;               
        while(stepSize >= 1)  
            for m = -stepSize : stepSize : stepSize        
                for n = -stepSize : stepSize : stepSize
                    refBlkVer = y + m;   
                    refBlkHor = x + n;  
                    if ( refBlkVer < 1 || refBlkVer+macroBlockSize-1 > row ...
                        || refBlkHor < 1 || refBlkHor+macroBlockSize-1 > col)
                        continue;
                    end
                    costRow = m/stepSize + 2;
                    costCol = n/stepSize + 2;
                    if (costRow == 2 && costCol == 2)
                        continue
                    end
                    costs(costRow, costCol ) = costFuncMAD(imgP(i:i+macroBlockSize-1,j:j+macroBlockSize-1), ...
                        imgI(refBlkVer:refBlkVer+macroBlockSize-1, refBlkHor:refBlkHor+macroBlockSize-1), macroBlockSize);
                    
                    computations = computations + 1;
                end
            end
        
            [dx, dy, min] = minCost(costs);

            x = x + (dx-2)*stepSize;
            y = y + (dy-2)*stepSize;
            stepSize = stepSize / 2;
            costs(2,2) = costs(dy,dx);
            
        end
        vectors(1,macroBlockCount) = y - i; 
        vectors(2,macroBlockCount) = x - j;            
        macroBlockCount = macroBlockCount + 1;
        costs = ones(3,3) * 65537;
    end
end
motionVector = vectors;
TSScomputations = computations/(macroBlockCount - 1);
end