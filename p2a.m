image = imread('EE569_2024Spring_HW4_materials/composite.png'); 

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

for i = 1:numFilters
    filterResponse = abs(conv2(double(image), filterBank{i}, 'same'));
    energyFeatures(:, :, i) = filterResponse;
end


windowSize = 10; 
energyMap = zeros(numRows, numCols, numFilters);
for i = 1:numFilters
    energyMap(:, :, i) = conv2(energyFeatures(:, :, i).^2, ones(windowSize)/(windowSize^2), 'same');
end


L5TL5_energy = energyMap(:, :, 1); 
normalizedEnergyMap = energyMap;
for i = 1:numFilters
    if i ~= 1 
        normalizedEnergyMap(:, :, i) = energyMap(:, :, i) ./ (L5TL5_energy + eps);
    end
end

numTextures = 5; 
normalizedEnergyMap24 = normalizedEnergyMap(: , : , 2:end);
features = reshape(normalizedEnergyMap, [], numFilters);
[~, centroids] = kmeans(features, numTextures, 'MaxIter', 2000); 
centroids = double(centroids);

idx = dsearchn(centroids, features); 
segmented_image = reshape(idx, numRows, numCols); 

figure;
imagesc(segmented_image);
colormap(jet(numTextures)); 
colorbar;
title('Segmented Image');

