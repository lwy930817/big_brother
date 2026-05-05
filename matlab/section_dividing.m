 clear;clear all;
    % 图像边界分析与分层处理
    % 输出:
    %   originalImage - 输入原图
    %   boundaryImage - 边界图
    %   regionMap - 检索区域分层图
    %   partitionMap - 重新分区图
    
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
%%
   
% Sobel边界检测(函数调用)
    boundaryImage = sobelBoundaryDetection(originalImage);
    
% 获取调制区域和分层编号(函数调用)
    [regionMap, layerInfo] = getModulationRegion(originalImage, boundaryImage);
    
% 对每层进行圆分析并重新分区
    [partitionMap, partitionInfo] = analyzeLayersWithCircle(regionMap, layerInfo, originalImage);

% 对每层进行圆分析并重新分区
     displayResults(originalImage, boundaryImage, regionMap, layerInfo, partitionMap, partitionInfo);
  %%
    function boundaryImage = sobelBoundaryDetection(grayImage)
    % Sobel边界检测
    % 使用Sobel算子检测图像边界
    
    % 应用Sobel算子
    sobelX = [-1 0 1; -2 0 2; -1 0 1];
    sobelY = [-1 -2 -1; 0 0 0; 1 2 1];
    
    Gx = imfilter(double(grayImage), sobelX, 'replicate');
    Gy = imfilter(double(grayImage), sobelY, 'replicate');
    
    % 计算梯度幅值
    gradientMagnitude = sqrt(Gx.^2 + Gy.^2);
    
    % 二值化处理得到边界
    threshold = 0.01 * max(gradientMagnitude(:)); % 自适应阈值
    boundaryImage = gradientMagnitude > threshold;

    se_erode = strel('square', 2);  
    boundaryImage = imerode(boundaryImage, se_erode); % 结构之外的那部分
end
%%
function [regionMap, layerInfo] = getModulationRegion(originalImage, boundaryImage)
    % 获取调制区域并进行分层编号
    % 输入:
    %   originalImage - 原图像
    %   boundaryImage - 边界图像
    % 输出:
    %   regionMap - 分层区域图
    %   layerInfo - 分层信息
    
    [rows, cols] = size(originalImage);
    regionMap = zeros(rows, cols); % 0表示非调制区域
    
    % 获取边界像素坐标
    [boundaryY, boundaryX] = find(boundaryImage);
    
    if isempty(boundaryY)
        error('未检测到边界');
    end
    %%
        % 创建距离变换图
    distMap = bwdist(boundaryImage);
    halfBandwidth_modulation = 100;

    % 创建分层编号   
    layerInfo = cell(halfBandwidth_modulation*2+1, 1); % 存储每层像素信息
     
    % 对调制区域进行分层
    for layer = 1:halfBandwidth_modulation*2+1
        % 当前层的距离范围
        if layer <= halfBandwidth_modulation
            % 内部层：距离边界 21-layer 
            innerDist = halfBandwidth_modulation+1 - layer;
            layerMask = (distMap >= innerDist) & (distMap < innerDist+1) & originalImage ;
        else
            % 外部层：距离边界 layer-20 
            outerDist = layer - halfBandwidth_modulation-1;
            layerMask = (distMap >= outerDist) & (distMap < outerDist+1) &~ originalImage;
            if outerDist==0
                layerMask=boundaryImage;
            end
        end
        % 为当前层分配编号
        regionMap(layerMask) = layer;
        
        % 存储当前层的信息
        [layerY, layerX] = find(layerMask);
        layerInfo{layer} = struct(...
            'layerNumber', layer, ...
            'pixelCount', sum(layerMask(:)), ...
            'pixelCoordinates', [layerX, layerY], ...
            'pixelValues', originalImage(layerMask) ...
        );
        
        fprintf('第%d层: %d个像素\n', layer, sum(layerMask(:)));
    end
end
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
%%
function circleMask = createCircleMask(radius)
    % 创建圆形掩膜
    diameter = 2 * radius + 1;
    [xx, yy] = meshgrid(1:diameter, 1:diameter);
    center = radius + 1;
    circleMask = (xx - center).^2 + (yy - center).^2 <= radius^2;
end
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
function displayResults(originalImage, boundaryImage, regionMap, layerInfo, partitionMap, partitionInfo)
    % 显示处理结果
    
    figure('Position', [100, 100, 1400, 1000], 'Name', '图像边界分析与分层分区结果');
    
    % 显示原图
    subplot(2, 3, 1);
    imshow(originalImage);
    title('输入原图', 'FontSize', 12, 'FontWeight', 'bold');
    colorbar;
    
    % 显示边界图
    subplot(2, 3, 2);
    imshow(boundaryImage);
    title('Sobel边界检测结果', 'FontSize', 12, 'FontWeight', 'bold');
    % imwrite(boundaryImage);
    % 显示分层区域图
    subplot(2, 3, 3);
    imagesc(regionMap);
    axis image;
    colormap(jet(41)); % 40层 + 背景
    colorbar;
    title('分层调制区域 (共%d层)', halfBandwidth_modulation,'FontSize', 12, 'FontWeight', 'bold');
    xlabel('内部 → 外部');
    %%
    % 显示重新分区图
    subplot(2, 3, 4);
    imagesc(partitionMap);
    axis image;
    colormap(jet(20));
    colorbar;
    caxis([0.5, 20.5]);
    title('圆分析重新分区 (20个区间)', 'FontSize', 12, 'FontWeight', 'bold');
    
    % 显示叠加图
    subplot(2, 3, 5);
    imshow(originalImage);
    hold on;
    
    % 创建彩色分区覆盖图
    coloredPartition = label2rgb(partitionMap, 'jet', 'k', 'shuffle');
    h = imshow(coloredPartition);
    set(h, 'AlphaData', partitionMap > 0 * 0.4); % 半透明显示
    
    % 绘制边界
    boundary = bwboundaries(boundaryImage);
    for k = 1:length(boundary)
        plot(boundary{k}(:,2), boundary{k}(:,1), 'r-', 'LineWidth', 2);
    end
    
    title('原图 + 重新分区 + 边界', 'FontSize', 12, 'FontWeight', 'bold');
    hold off;
    
    % 显示分区统计饼图
    subplot(2, 3, 6);
    partitionCounts = cellfun(@(x) x.pixelCount, partitionInfo);
    partitionNumbers = 1:20;
    
    % 创建折线图
    plot(partitionNumbers, partitionCounts, 'b-o', 'LineWidth', 2, 'MarkerSize', 6, 'MarkerFaceColor', 'b');
    grid on;
    
    % 设置坐标轴标签和标题
    xlabel('分区编号', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('像素个数', 'FontSize', 12, 'FontWeight', 'bold');
    title('各分区像素分布', 'FontSize', 12, 'FontWeight', 'bold');
    
    % 设置x轴刻度
    xticks(1:20);
    
    % 在数据点上添加数值标签
    for i = 1:20
        text(partitionNumbers(i), partitionCounts(i), ...
            sprintf('%d', partitionCounts(i)), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
            'FontSize', 9, 'FontWeight', 'bold');
    end
    
    % 显示统计信息
    fprintf('\n=== 分层统计信息 ===\n');
    fprintf('总层数: %d\n', length(layerInfo));
    fprintf('调制区域总像素数: %d\n', sum(regionMap(:) > 0));
    
    totalPixels = numel(originalImage);
    modulationRatio = sum(regionMap(:) > 0) / totalPixels * 100;
    fprintf('调制区域占比: %.2f%%\n', modulationRatio);
    
    % 显示每层的基本信息
    fprintf('\n=== 各层详细信息 ===\n');
    for i = 1:length(layerInfo)
        info = layerInfo{i};
        if i <= 20
            position = '内部';
            distance = 21 - i;
        else
            position = '外部';
            distance = i - 20;
        end
        fprintf('层 %2d (%s%d): %5d像素, 平均灰度: %6.1f\n', ...
            i, position, distance, info.pixelCount, mean(info.pixelValues));
    end
    
    % 显示分区统计信息
    fprintf('\n=== 圆分析重新分区统计 ===\n');
    fprintf('分区区间\t像素数量\t占比\t\t平均灰度值\n');
    for i = 1:20
        info = partitionInfo{i}; 
        if info.pixelCount > 0
            ratio = info.pixelCount / sum(partitionCounts) * 100;
            avgGray = mean(info.pixelValues);
            fprintf('%.1f-%.1f\t\t%d\t\t%.1f%%\t\t%.1f\n', ...
                (i-1)*0.05, i*0.05, info.pixelCount, ratio, avgGray);
        else 
            fprintf('%.1f-%.1f\t\t%d\t\t0.0%%\t\tN/A\n', (i-1)*0.1, i*0.1, info.pixelCount);
        end
    end
    
    % 显示各分区中的层分布
    fprintf('\n=== 各分区中的层分布 ===\n');
    fprintf('分区\\层\t');
    for layer = 1:length(layerInfo)
        fprintf('%2d\t', layer);
    end
    fprintf('\n');
    
    for i = 1:20
        fprintf('%.1f-%.1f\t', (i-1)*0.05, i*0.05);
        for layer = 1:length(layerInfo)
            count = partitionInfo{i}.layerDistribution(layer);
            if count > 0
                fprintf('%2d\t', count);
            else
                fprintf('  \t');
            end
        end
        fprintf('\n');
    end
end