clear; close all; clc;
%%
% 选择并读取图像
[filename, pathname] = uigetfile({'*.jpg;*.png;*.bmp;*.tif', '图像文件 (*.jpg, *.png, *.bmp, *.tif)'}, '选择灰度图像');
if isequal(filename, 0)
    error('未选择图像文件');
end

imagePath = fullfile(pathname, filename);
originalImage = imread(imagePath);
% 转换为灰度图像（如果输入是彩色图像）
if size(originalImage, 3) == 3
    originalImage = rgb2gray(originalImage);
end 

% 确保图像是uint8类型
originalImage = im2uint8(originalImage);
[rows, cols] = size(originalImage);
% ========== 边界检测（Sobel） ==========
boundaryImage = sobelBoundaryDetection(originalImage);
    
% 对每层进行圆分析并重新分区（此处只对中间层，即边界分区）
    [partitionMap, partitionInfo] = analyzeLayersWithCircle(regionMap, layerInfo, originalImage);

% 分层-分区-及补偿
halfBandList = 20:20:60;          % 带宽：20/40/60
% compensationValues = 17:17:255;   % 补偿灰度：17~255，公差17
for k = 1:length(halfBandList)
    halfBandwidth_modulation = halfBandList(k);
    fprintf('\n========== 当前半带宽：%d ==========\n',halfBandwidth_modulation);
 % ========== 1. 原有逻辑：边界检测 + 分层 ==========

    [regionMap, layerInfo] = getModulationRegion(originalImage, boundaryImage, halfBandwidth_modulation);


end

 



%% Sobel边界检测
function boundaryImage = sobelBoundaryDetection(grayImage)
    sobelX = [-1 0 1; -2 0 2; -1 0 1];
    sobelY = [-1 -2 -1; 0 0 0; 1 2 1];
    
    Gx = imfilter(double(grayImage), sobelX, 'replicate');
    Gy = imfilter(double(grayImage), sobelY, 'replicate');
    
    gradientMagnitude = sqrt(Gx.^2 + Gy.^2);
    threshold = 0.01 * max(gradientMagnitude(:));
    boundaryImage = gradientMagnitude > threshold;

    se_erode = strel('square', 2);  
    boundaryImage = imerode(boundaryImage, se_erode);
end

%% 分层与其补偿逻辑
function [regionMap, layerInfo] = getModulationRegion(originalImage, boundaryImage, halfBandwidth_modulation)
    [rows, cols] = size(originalImage);
    regionMap = zeros(rows, cols);
    
    [boundaryY, boundaryX] = find(boundaryImage);
    if isempty(boundaryY)
        error('未检测到边界');
    end

    distMap = bwdist(boundaryImage);
    layerInfo = cell(halfBandwidth_modulation*2+1, 1);

    for layer = 1:halfBandwidth_modulation*2+1
        if layer <= halfBandwidth_modulation
            innerDist = halfBandwidth_modulation+1 - layer;
            layerMask = (distMap >= innerDist) & (distMap < innerDist+1) & originalImage ;
        else
            outerDist = layer - halfBandwidth_modulation-1;
            layerMask = (distMap >= outerDist) & (distMap < outerDist+1) &~ originalImage;
            if outerDist==0
                layerMask=boundaryImage;
            end
        end
        regionMap(layerMask) = layer;
        
        [layerY, layerX] = find(layerMask);
        layerInfo{layer} = struct(...
            'layerNumber', layer, ...
            'pixelCount', sum(layerMask(:)), ...
            'pixelCoordinates', [layerX, layerY], ...
            'pixelValues', originalImage(layerMask) ...
        );
        
        fprintf('第%d层: %d个像素\n', layer, sum(layerMask(:)));


        % for comp = compensationValues
        % ---------------- 层间灰度调制（咱设定为满灰度）  ----------------
        comp=255;
        % 复制原图
        compensatedImage = double(originalImage); 
        % 仅分层区域叠加补偿灰度
        compensatedImage(regionMap > 0) = compensatedImage(regionMap > 0) + comp;
        % 限制最高灰度为255，防止溢出
        compensatedImage = min(compensatedImage, 255);
        % 转回uint8图像格式
        compensatedImage = uint8(compensatedImage);

        % 保存图像
        % savePath = fullfile(outputFolder, sprintf('补偿灰度_%03d.png', comp));
        % imwrite(compensatedImage, savePath);
        % 
        % fprintf('已保存：%s | 补偿灰度：%d\n', savePath, comp);
        % end
    end
end
% 补偿逻辑
%%
function [partitionMap, partitionInfo] = analyzeLayersWithCircle(regionMap, layerInfo, originalImage)
    % 对每层进行圆分析并重新分区
    % 输入:
    %   regionMap - 分层区域图
    %   layerInfo - 分层信息
    %   originalImage - 原图像
    % 输出:
    %   partitionMap - 重新分区图 (1-20表示10个比值区间)
    %   partitionInfo - 分区信息
    
    fprintf('\n开始圆分析分区处理...\n');
    
    [rows, cols] = size(originalImage);
    partitionMap = zeros(rows, cols); % 0表示非调制区域
    
    % 创建直径为15像素的圆形掩膜
    circleRadius = 14; % 半径=7，直径=15,PSF小于2um，映射到DMD为15.39
    circleMask = createCircleMask(circleRadius);
    circleTotalPixels = sum(circleMask(:));
    
    fprintf('圆形掩膜直径: %d像素, 总面积: %d像素\n', 2*circleRadius+1, circleTotalPixels);
    
    % 预处理：为每层创建填充后的掩膜
    numLayers = length(layerInfo);
    filledLayerMasks = cell(numLayers, 1);
    
    for layer = 1:numLayers
        layerMask = (regionMap == layer);
        img=layerMask;
        % 填充内部空隙
        if islogical(img)
    % 已经是二值图（logical），直接使用（1=白色边界，0=黑色背景/内孔）
              binary_img = img;
         else
    % 非二值图：先转灰度→二值化
           if size(img, 3) == 3
              img_gray = rgb2gray(img);  % 彩色图转灰度
    else
             img_gray = img;  % 已为灰度图
    end
             binary_img =imbinarize(img_gray);  % 仅对灰度图二值化
        end
        boundaries = bwboundaries(binary_img, 'noholes');
      
        if isempty(boundaries)
    error('未检测到闭合外边界，请检查图像');
end
        boundary_areas = cellfun(@(b) polyarea(b(:,2), b(:,1)), boundaries);
        [~, max_boundary_idx] = max(boundary_areas);
        outer_boundary = boundaries{max_boundary_idx};
        
        % 修正边界坐标（覆盖图像边缘，强制闭合）
        outer_boundary(:,1) = max(min(outer_boundary(:,1), rows), 1);
        outer_boundary(:,2) = max(min(outer_boundary(:,2), cols), 1);
        if ~isequal(outer_boundary(1,:), outer_boundary(end,:))
            outer_boundary = [outer_boundary; outer_boundary(1,:)];
        end
        
        % 生成外边界掩码+内孔掩码+带状掩膜
        outer_mask = poly2mask(outer_boundary(:,2), outer_boundary(:,1), rows, cols);
        inner_region_img = binary_img & outer_mask;
        inner_fill_all = imfill(inner_region_img, 'holes');
        inner_hole_mask = inner_fill_all & ~inner_region_img;
        band_fill_mask = outer_mask & ~inner_hole_mask;  % 带状填充掩膜（1=带状区域）
        
        % ===================== 核心：计算带状掩膜与原始边界的并集 =====================
        % 逻辑或（|）= 并集：保留带状掩膜（band_fill_mask）和原始边界（binary_img）的所有1像素
        filledLayerMasks{layer} = band_fill_mask | binary_img;  % 最终并集结果（1=带状区域+原始边界）
        
        % 4. 提取最终带状区域（外边界内区域 - 内孔区域）
        % filledLayerMasks{layer} = outer_mask & ~inner_hole_mask;  % 1=带状区，0=背景/内孔
        % filledLayerMasks{layer}=union(filledLayerMasks{layer}, outer_boundary);
        % filledLayerMasks{layer}=filledLayerMasks{layer}|boundarie;
        % filledLayerMasks{layer}
        % filledLayerMasks{layer} = imfill(img, 'holes');
    end
       
    
    % 初始化分区信息
    partitionInfo = cell(20, 1);
    for i = 1:20
        partitionInfo{i} = struct(...
            'partitionNumber', i, ...
            'ratioRange', [(i-1)*0.05, i*0.05], ...
            'pixelCount', 0, ...
            'pixelCoordinates', [], ...
            'pixelValues', [], ...
            'layerDistribution', zeros(numLayers, 1) ...
        );
    end
       
    % 对每层的每个像素进行圆分析
    totalPixels = sum(cellfun(@(x) x.pixelCount, layerInfo));
    processedPixels = 0;

    for layer = 1:numLayers
        fprintf('处理第%d层...\n', layer);
        
        layerPixels = layerInfo{layer}.pixelCoordinates;
        layerPixelCount = size(layerPixels, 1);
        filledMask = filledLayerMasks{layer};
        
        for i = 1:layerPixelCount
            x = layerPixels(i, 1);
            y = layerPixels(i, 2);
            
            % 计算圆形区域与当前层填充掩膜的交集
            intersectionRatio = calculateCircleIntersection(x, y, circleMask, filledMask, rows, cols);
            
            % 根据比值确定分区 (1-20)
            partitionLevel = min(20, max(1, ceil(intersectionRatio * 20)));
            if intersectionRatio == 0
                partitionLevel = 1; % 比值为0的归入第一个区间
            end
            
            % 更新分区图
            partitionMap(y, x) = partitionLevel;
            
            % 更新分区信息
            partitionInfo{partitionLevel}.pixelCount = partitionInfo{partitionLevel}.pixelCount + 1;
            partitionInfo{partitionLevel}.pixelCoordinates = [partitionInfo{partitionLevel}.pixelCoordinates; [x, y]];
            partitionInfo{partitionLevel}.pixelValues = [partitionInfo{partitionLevel}.pixelValues; originalImage(y, x)];
            partitionInfo{partitionLevel}.layerDistribution(layer) = partitionInfo{partitionLevel}.layerDistribution(layer) + 1;
            
            processedPixels = processedPixels + 1;
            if mod(processedPixels, 1000) == 0
                fprintf('已处理: %d/%d 像素 (%.1f%%)\n', processedPixels, totalPixels, processedPixels/totalPixels*100);
            end
        end
    end
    
    fprintf('圆分析分区处理完成!\n');
end
%% 圆掩膜

function circleMask = createCircleMask(radius)
    radius = 15;
%%
    % 创建【平滑抗锯齿边界】圆形掩膜（效果接近AI生成圆）
    % 输入：radius - 圆的半径（整数，如7/13/26）
    % 输出：circleMask - 平滑圆掩膜（double类型，值范围[0,1]）

    % 1. 基础尺寸设置（和原函数完全一致，保证兼容性）
    diameter = 2 * radius + 1;
    center = radius + 1;  % 圆心坐标（像素中心）

    % 2. 生成像素坐标网格
    [xx, yy] = meshgrid(1:diameter, 1:diameter);

    % 3. 计算每个像素中心到圆心的欧氏距离
    dist = sqrt((xx - center).^2 + (yy - center).^2);

    % 4. 初始化掩膜（全黑，值为0）
    circleMask = zeros(diameter, 'double');

    % 5. 填充实心部分（完全不透明，值为1）
    solidRegion = dist <= radius - 0.5;
    circleMask(solidRegion) = 1.0;

    % 6. 边界平滑过渡（核心优化：0.5像素线性渐变，消除锯齿）
    edgeRegion = (dist > radius - 0.5) & (dist < radius + 0.5);
    % 计算过渡权重：从1线性降到0，实现边缘柔化
    circleMask(edgeRegion) = 1 - (dist(edgeRegion) - (radius - 0.5));

    % 7. 显示平滑圆（灰度图，清晰看到边缘过渡）
    figure('Color','w');
    imshow(circleMask);
    colormap(gca, gray);  % 用灰度图展示边缘过渡效果
    axis equal; axis off;
    title(sprintf('半径=%d 平滑圆掩膜（抗锯齿边界）', radius), 'FontSize', 12);
    %%
end
% function circleMask = createCircleMask(radius)
%     % 创建圆形掩膜
%     diameter = 2 * radius + 1;
%     [xx, yy] = meshgrid(1:diameter, 1:diameter);
%     center = radius + 1;
%     circleMask = (xx - center).^2 + (yy - center).^2 <= radius^2;
%     end
%%
function intersectionRatio = calculateCircleIntersection(x, y, circleMask, filledMask, rows, cols)
    % 计算圆形区域与填充掩膜的交集比值
    circleSize = size(circleMask, 1);
    radius = (circleSize - 1) / 2;
    
    % 计算圆形区域在图像中的位置
    xStart = x - radius;
    xEnd = x + radius;
    yStart = y - radius;
    yEnd = y + radius;
    
    % 处理边界情况
    circlePatch = circleMask;
    maskPatch = false(circleSize, circleSize);
    
    % 提取填充掩膜的对应区域
    xStartImg = max(1, xStart);
    xEndImg = min(cols, xEnd);
    yStartImg = max(1, yStart);
    yEndImg = min(rows, yEnd);
    
    % 计算在圆形掩膜中的对应区域
    xStartCircle = max(1, 1 - (xStart - 1));
    xEndCircle = min(circleSize, circleSize - (xEnd - cols));
    yStartCircle = max(1, 1 - (yStart - 1));
    yEndCircle = min(circleSize, circleSize - (yEnd - rows));
    
    % 提取掩膜区域
    maskPatch(yStartCircle:yEndCircle, xStartCircle:xEndCircle) = ...
        filledMask(yStartImg:yEndImg, xStartImg:xEndImg);
    
    % 计算交集
    intersectionMask = circlePatch & maskPatch;
    intersectionPixels = sum(intersectionMask(:));
    circlePixelsInImage = sum(circlePatch(:)); % 实际在图像内的圆形像素数
    
    if circlePixelsInImage == 0
        intersectionRatio = 0;
    else
        intersectionRatio = intersectionPixels / circlePixelsInImage;
    end
end

%%
% 画布尺寸
W = 1920;
H = 1080;

% 圆参数
d =52;
r = d/2;
cx = W/2;
cy = H/2;

% 1. 生成网格（亚像素采样）
[x, y] = meshgrid(1:W, 1:H);

% 2. 计算每个像素到圆心的距离
dist = sqrt((x - cx).^2 + (y - cy).^2);

% 3. 做一个"软边圆"：边缘1像素过渡（交错效果）
% 半径 r-0.5 到 r+0.5 之间做线性渐变
img = zeros(H, W, 'uint8');
% 中心实心部分（白色）
img(dist <= r - 0.5) = 255;
% 边缘过渡部分（0~255渐变）
edge_mask = (dist > r - 0.5) & (dist < r + 0.5);
img(edge_mask) = uint8(255 * (1 - (dist(edge_mask) - (r - 0.5))));

% 4. 显示
figure;
imshow(img);
title('1920×1080 黑色背景，抗锯齿直径106像素圆');










 %%
% % ========== 参数设置 ==========
% % 分层选项：40、60、80（内部+外部层数，边界单独一层）
% layerOptions = [20, 30, 40];  % 对应的内部/外部层数，总层数 = 2*layerOptions + 1
% layerChoice = listdlg('PromptString','选择分层数（内部+外部层数）:',...
%     'SelectionMode','single','ListString',{'40层 (内部20+边界1+外部20)','60层 (内部30+边界1+外部30)','80层 (内部40+边界1+外部40)'});
% if isempty(layerChoice), return; end
% innerOuterLayers = layerOptions(layerChoice);
% totalLayers = 2*innerOuterLayers + 1;  % 总层数（包含边界）
% 
% % 圆掩膜直径范围：13 ~ 65 像素（奇数）
% diameterVec = 13:13:65;
% % 曲率阈值（度），大于该值判定为拐角/圆角
% curvatureThreshold = 30;
% 
% % ========== 读取图像 ==========
% [filename, pathname] = uigetfile({'*.jpg;*.png;*.bmp;*.tif', '图像文件'}, '选择灰度图像');
% if isequal(filename, 0), error('未选择图像文件'); end
% originalImage = imread(fullfile(pathname, filename));  
% if size(originalImage,3)==3, originalImage = rgb2gray(originalImage); end
% originalImage = im2uint8(originalImage);
% [rows, cols] = size(originalImage);
% 
% % ========== 边界检测（Sobel，与原逻辑一致） ==========
% boundaryImage = sobelBoundaryDetection(originalImage);
% 
% % ========== 二值化目标区域（结构区域） ==========
% objectMask = imbinarize(originalImage);   % 原始灰度图二值化，得到物体区域
% 
% % ========== 分层处理（与原逻辑相同，支持可变层数） ==========
% [regionMap, layerInfo] = getModulationRegion(originalImage, boundaryImage, innerOuterLayers);
% 
% % ========== 提取边界轮廓点 ==========
% boundaryPts = bwboundaries(boundaryImage);
% if isempty(boundaryPts), error('未检测到边界轮廓'); end
% % 取最长轮廓（假设主轮廓）
% boundaryPts = boundaryPts{1};  % N x 2, [y, x]
% boundaryX = boundaryPts(:,2);
% boundaryY = boundaryPts(:,1);
% numBoundaryPts = length(boundaryX);
% 
% % ========== 计算每个边界点的曲率（判断直线/拐角） ==========
% % 计算相邻点向量及夹角变化
% angles = zeros(numBoundaryPts,1);
% for i = 1:numBoundaryPts
%     prevIdx = mod(i-2, numBoundaryPts) + 1;
%     nextIdx = mod(i, numBoundaryPts) + 1;
%     v1 = [boundaryX(i)-boundaryX(prevIdx), boundaryY(i)-boundaryY(prevIdx)];
%     v2 = [boundaryX(nextIdx)-boundaryX(i), boundaryY(nextIdx)-boundaryY(i)];
%     % 夹角（弧度转度）
%     cosTheta = dot(v1,v2) / (norm(v1)*norm(v2)+eps);
%     angles(i) = acosd(max(-1,min(1,cosTheta)));
% end
% isCorner = angles > curvatureThreshold;   % 曲率大视为拐角/圆角
% 
% % ========== 距离变换：存储每个像素最近边界点的索引 ==========
% dist2boundary = bwdist(boundaryImage);
% % 为了获得最近边界点的索引，使用 bwdist 的第二个输出
% [~, idxMap] = bwdist(boundaryImage);  % idxMap 中每个像素存储最近边界点的线性索引
% % 将线性索引转换为边界点序号（需建立映射）
% boundaryLinearIdx = sub2ind([rows,cols], boundaryY, boundaryX);
% idx2pt = zeros(rows*cols,1);
% idx2pt(boundaryLinearIdx) = 1:numBoundaryPts;
% % 为每个像素获取对应边界点的序号
% nearestBoundaryIdx = idx2pt(idxMap(:));
% nearestBoundaryIdx = reshape(nearestBoundaryIdx, rows, cols);
% % 仅调制区域内有效（regionMap>0）
% modulationMask = regionMap > 0;
% nearestBoundaryIdx(~modulationMask) = 0;
% 
% % ========== 对不同直径的圆掩膜分别处理并保存补偿图像 ==========
% outputDir = 'compensation_results';
% if ~exist(outputDir,'dir'), mkdir(outputDir); end
% 
% for d = diameterVec
%     radius = floor(d/2);
%     fprintf('\n===== 处理圆直径: %d 像素 (半径 %d) =====\n', d, radius);
% 
%     % 创建圆形掩膜（直径为 d）
%     circleMask = createCircleMask(radius);
%     circleArea = sum(circleMask(:));
% 
%     % 1. 计算每个边界点的圆内目标区域占比 p
%     pValue = zeros(numBoundaryPts,1);
%     for i = 1:numBoundaryPts
%         cx = boundaryX(i);
%         cy = boundaryY(i);
%         % 计算圆形区域内的目标区域占比
%         pValue(i) = computeCircleObjectRatio(cx, cy, circleMask, objectMask, rows, cols);
%     end
% 
%     % 2. 计算每个边界点的补偿强度系数 w = 1 - 2*p
%     w = 1 - 2 * pValue;
%     w = max(0, min(1, w));   % 限制在 [0,1]
%     % 对于拐角点，补偿强度可以额外增强（这里直接使用 w；若需要区别对待可修改）
%     strength = w;   % 边界点处的原始补偿强度
% 
%     % 3. 将边界点的强度传播到周围像素：距离衰减因子 (1 - d/maxLayer)
%     maxLayer = innerOuterLayers;   % 最大影响距离（内部/外部层数）
%     % 计算每个调制像素到其最近边界点的距离（已有 dist2boundary）
%     distMap = bwdist(boundaryImage);
%     % 衰减因子：距离越近影响越大，线性衰减至0
%     decay = max(0, 1 - distMap / maxLayer);
%     decay(~modulationMask) = 0;
% 
%     % 根据每个像素的最近边界点索引，获取该边界点的强度
%     compStrength = zeros(rows,cols);
%     for i = 1:numBoundaryPts
%         mask = (nearestBoundaryIdx == i);
%         compStrength(mask) = strength(i);
%     end
%     % 最终补偿系数 = 衰减因子 * 对应边界点强度
%     compensationCoeff = decay .* compStrength;
% 
%     % 4. 应用补偿：补偿值 = 补偿系数 * 灰度范围（例如 128），叠加到原图
%     compensationValue = compensationCoeff * 128;   % 可调整幅度
%     compensatedImage = double(originalImage) + compensationValue;
%     compensatedImage = uint8(min(255, max(0, compensatedImage)));
% 
%     % 5. 保存结果，命名：总层数-圆直径.bmp
%     outputName = sprintf('%d-%d.bmp', totalLayers, d);
%     outputPath = fullfile(outputDir, outputName);
%     imwrite(compensatedImage, outputPath);
%     fprintf('已保存: %s\n', outputPath);
% 
%     % 可选：显示第一个直径的补偿结果
%     if d == diameterVec(1)
%         figure('Name', sprintf('补偿示例 (总层数=%d, 直径=%d)', totalLayers, d));
%         subplot(1,2,1); imshow(originalImage); title('原始图像');
%         subplot(1,2,2); imshow(compensatedImage); title('补偿后图像');
%     end
% end
% 
% fprintf('\n所有补偿图像已保存至目录: %s\n', outputDir);
% 
% % ==================== 以下为所有子函数 ====================
% 
% function boundaryImage = sobelBoundaryDetection(grayImage)
%     % Sobel边界检测（与原代码完全一致）
%     sobelX = [-1 0 1; -2 0 2; -1 0 1];
%     sobelY = [-1 -2 -1; 0 0 0; 1 2 1];
%     Gx = imfilter(double(grayImage), sobelX, 'replicate');
%     Gy = imfilter(double(grayImage), sobelY, 'replicate');
%     gradientMagnitude = sqrt(Gx.^2 + Gy.^2);
%     threshold = 0.01 * max(gradientMagnitude(:));
%     boundaryImage = gradientMagnitude > threshold;
%     se_erode = strel('square', 2);
%     boundaryImage = imerode(boundaryImage, se_erode);
% end
% 
% function [regionMap, layerInfo] = getModulationRegion(originalImage, boundaryImage, halfBandwidth)
%     % 分层处理（支持可变半带宽）
%     [rows, cols] = size(originalImage);
%     regionMap = zeros(rows, cols);
%     distMap = bwdist(boundaryImage);
%     totalLayers = 2*halfBandwidth + 1;
%     layerInfo = cell(totalLayers, 1);
% 
%     for layer = 1:totalLayers
%         if layer <= halfBandwidth   % 内部层
%             innerDist = halfBandwidth + 1 - layer;
%             layerMask = (distMap >= innerDist) & (distMap < innerDist+1) & (originalImage > 0);
%         elseif layer == halfBandwidth+1  % 边界层
%             layerMask = boundaryImage;
%         else                         % 外部层
%             outerDist = layer - halfBandwidth - 1;
%             layerMask = (distMap >= outerDist) & (distMap < outerDist+1) & ~(originalImage > 0);
%             if outerDist == 0
%                 layerMask = boundaryImage;
%             end
%         end
%         regionMap(layerMask) = layer;
%         [layerY, layerX] = find(layerMask);
%         layerInfo{layer} = struct(...
%             'layerNumber', layer, ...
%             'pixelCount', sum(layerMask(:)), ...
%             'pixelCoordinates', [layerX, layerY], ...
%             'pixelValues', originalImage(layerMask));
%         % fprintf('第%d层: %d个像素\n', layer, sum(layerMask(:)));
%     end
% end
% 
% function circleMask = createCircleMask(radius)
%     % 创建半径为 radius 的圆形掩膜（直径 = 2*radius+1）
%     diameter = 2*radius + 1;
%     [xx, yy] = meshgrid(1:diameter, 1:diameter);
%     center = radius + 1;
%     circleMask = (xx - center).^2 + (yy - center).^2 <= radius^2;
% end
% 
% function ratio = computeCircleObjectRatio(cx, cy, circleMask, objectMask, rows, cols)
%     % 计算以 (cx,cy) 为中心，圆形掩膜内 objectMask 的面积占比
%     radius = (size(circleMask,1)-1)/2;
%     xStart = cx - radius; xEnd = cx + radius;
%     yStart = cy - radius; yEnd = cy + radius;
%     % 裁剪到图像边界
%     xStartImg = max(1, ceil(xStart));
%     xEndImg = min(cols, floor(xEnd));
%     yStartImg = max(1, ceil(yStart));
%     yEndImg = min(rows, floor(yEnd));
%     % 在掩膜中的对应区域
%     xStartMask = max(1, xStartImg - xStart + 1);
%     xEndMask = size(circleMask,2) - (xEnd - xEndImg);
%     yStartMask = max(1, yStartImg - yStart + 1);
%     yEndMask = size(circleMask,1) - (yEnd - yEndImg);
%     if xStartMask > xEndMask || yStartMask > yEndMask
%         ratio = 0;
%         return;
%     end
%     % 提取有效圆形区域
%     validCircle = circleMask(yStartMask:yEndMask, xStartMask:xEndMask);
%     validObject = objectMask(yStartImg:yEndImg, xStartImg:xEndImg);
%     intersectionPixels = sum(validCircle(:) & validObject(:));
%     totalCirclePixels = sum(validCircle(:));
%     if totalCirclePixels == 0
%         ratio = 0;
%     else
%         ratio = intersectionPixels / totalCirclePixels;
%     end
% end