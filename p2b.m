image = imread('EE569_2024Spring_HW4_materials/composite.png'); % 读取图像，假设图像已经存储在当前目录下

if size(image, 3) == 3
    image = rgb2gray(image);
end

L5 = [1 4 6 4 1];
E5 = [-1 -2 0 2 1];
S5 = [-1 0 2 0 -1];
W5 = [-1 2 0 -2 1];
R5 = [1 -4 6 -4 1];

filterBank = {
    conv2(L5', L5), conv2(L5', E5), conv2(L5', S5), conv2(L5', W5), conv2(L5', R5), conv2(E5', L5), conv2(E5', E5), conv2(E5', S5), conv2(E5', W5), conv2(E5', R5), conv2(S5', L5), conv2(S5', E5), conv2(S5', S5), conv2(S5', W5), conv2(S5', R5), conv2(W5', L5), conv2(W5', E5), conv2(W5', S5), conv2(W5', W5), conv2(W5', R5), conv2(R5', L5), conv2(R5', E5), conv2(R5', S5), conv2(R5', W5), conv2(R5', R5)
};

[numRows, numCols] = size(image);
numFilters = length(filterBank);
energyFeatures = zeros(numRows, numCols, numFilters);

% 1. Filter bank response computation
for i = 1:numFilters
    filterResponse = abs(conv2(double(image), filterBank{i}, 'same'));
    energyFeatures(:, :, i) = filterResponse;
end

% 2. Energy feature computation
windowSize = 10; 
energyMap = zeros(numRows, numCols, numFilters);
for i = 1:numFilters
    energyMap(:, :, i) = conv2(energyFeatures(:, :, i).^2, ones(windowSize)/(windowSize^2), 'same');
end

% 3. Energy feature normalization
L5TL5_energy = energyMap(:, :, 1); 
normalizedEnergyMap = energyMap;
for i = 1:numFilters
    if i ~= 1 
        normalizedEnergyMap(:, :, i) = energyMap(:, :, i) ./ (L5TL5_energy + eps); 
    end
end

% PCA for feature reduction
numComponents = 10; 
numTextures = 5; 

features_flattened = reshape(normalizedEnergyMap, [], numFilters); 
coeff = pca(features_flattened); 
features_reduced = features_flattened * coeff(:, 1:numComponents); 


[~, centroids] = kmeans(features_reduced, numTextures, 'MaxIter', 2000); 
centroids = double(centroids);

idx = dsearchn(centroids, features_reduced); 
segmented_image = reshape(idx, numRows, numCols); 

minRegionSize = 10000; 
regionProps = regionprops(segmented_image, 'Area');
for i = 1:length(regionProps)
    if regionProps(i).Area < minRegionSize
        segmented_image(segmented_image == i) = mode(segmented_image(:)); 
    end
end

threshold = 0.5;
adjacentRegions = bwperim(segmented_image);

adjacentTextureFeatures = zeros(size(adjacentRegions, 1), numComponents);
for i = 1:size(adjacentRegions, 1)
    [row, col] = find(adjacentRegions(i, :));
    regionFeatures = features_reduced(sub2ind([numRows, numCols], row, col), :);
    adjacentTextureFeatures(i, :) = mean(regionFeatures, 1);
end

for i = 1:size(adjacentRegions, 1)
    [row, col] = find(adjacentRegions(i, :));
    for j = 1:length(row)
        regionIdx = segmented_image(row(j), col(j));
        
        adjacentTexture = adjacentTextureFeatures(i, :);
        
        currentTexture = features_reduced(sub2ind([numRows, numCols], row(j), col(j)), :);
        
        similarity = pdist2(currentTexture, adjacentTexture, 'euclidean');
        
        if similarity < threshold 
            segmented_image(row(j), col(j)) = mode(segmented_image(:));
        end
    end
end

figure;
imagesc(segmented_image);
colormap(jet(numTextures)); 
colorbar;
title('Segmented Image');


