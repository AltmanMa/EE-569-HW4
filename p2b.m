% 读取图像
image = imread('EE569_2024Spring_HW4_materials/composite.png'); % 读取图像，假设图像已经存储在当前目录下

% 转换图像为灰度图像
if size(image, 3) == 3
    image = rgb2gray(image);
end

% 初始化变量
[numRows, numCols] = size(image);
numFilters = length(filterBank);
energyFeatures = zeros(numRows, numCols, numFilters);

% 1. Filter bank response computation
for i = 1:numFilters
    filterResponse = abs(conv2(double(image), filterBank{i}, 'same'));
    energyFeatures(:, :, i) = filterResponse;
end

% 2. Energy feature computation
windowSize = 3; % 可以根据需要调整窗口大小
energyMap = zeros(numRows, numCols, numFilters);
for i = 1:numFilters
    energyMap(:, :, i) = conv2(energyFeatures(:, :, i).^2, ones(windowSize)/(windowSize^2), 'same');
end

% 3. Energy feature normalization
L5TL5_energy = energyMap(:, :, 21); % L5TL5滤波器的能量
normalizedEnergyMap = energyMap;
for i = 1:numFilters
    if i ~= 21 % 不考虑L5TL5滤波器
        normalizedEnergyMap(:, :, i) = energyMap(:, :, i) ./ (L5TL5_energy + eps); % 加上一个很小的数，防止除零错误
    end
end

% PCA for feature reduction
numComponents = 10; % 设置降维后的特征数量
features_flattened = reshape(normalizedEnergyMap, [], numFilters); % 修正这里的变量名
coeff = pca(features_flattened); % 计算 PCA 的主成分
features_reduced = features_flattened * coeff(:, 1:numComponents); % 使用前 numComponents 个主成分进行特征降维

% 使用降维后的特征进行纹理分割
[~, centroids] = kmeans(features_reduced, numTextures); % 使用K-means算法找到聚类中心
centroids = double(centroids);

idx = dsearchn(centroids, features_reduced); % 找到每个像素点所属的类别
segmented_image = reshape(idx, numRows, numCols); % 重新构造分割后的图像

minRegionSize = 100; % 定义最小区域大小，用于决定是否合并
regionProps = regionprops(segmented_image, 'Area');
for i = 1:length(regionProps)
    if regionProps(i).Area < minRegionSize
        segmented_image(segmented_image == i) = mode(segmented_image(:)); % 将小区域中的像素标记为与其相邻区域中的像素标记相同的标签
    end
end

% 显示分割结果
figure;
imagesc(segmented_image);
colormap(jet(numTextures)); % 使用jet colormap，分别表示不同的纹理
colorbar;
title('Segmented Image');

% 保存分割结果
imwrite(segmented_image, 'segmented_image.png');

