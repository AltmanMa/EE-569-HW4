folderPath = 'EE569_2024Spring_HW4_materials/test';
newFolderPath = 'test_energy';

rawFiles = dir(fullfile(folderPath, '*.raw'));
numImages = length(rawFiles);
allEnergyFeatures = zeros(numImages, 25);  

L5 = [1 4 6 4 1];
E5 = [-1 -2 0 2 1];
S5 = [-1 0 2 0 -1];
W5 = [-1 2 0 -2 1];
R5 = [1 -4 6 -4 1];

filterBank = {
    conv2(L5', L5), conv2(L5', E5), conv2(L5', S5), conv2(L5', W5), conv2(L5', R5), conv2(E5', L5), conv2(E5', E5), conv2(E5', S5), conv2(E5', W5), conv2(E5', R5), conv2(S5', L5), conv2(S5', E5), conv2(S5', S5), conv2(S5', W5), conv2(S5', R5), conv2(W5', L5), conv2(W5', E5), conv2(W5', S5), conv2(W5', W5), conv2(W5', R5), conv2(R5', L5), conv2(R5', E5), conv2(R5', S5), conv2(R5', W5), conv2(R5', R5)
};
idx = 1;


for file = rawFiles'
    filename = fullfile(folderPath, file.name);
    
    imageData = readraw(filename, 128, 128);
    
    filterResponses = zeros(128, 128, length(filterBank));

    for i = 1:length(filterBank)
        filter = filterBank{i};
        extendedImage = padarray(imageData, [2 2], 'replicate', 'both');
        filterResponses(:, :, i) = conv2(extendedImage, filter, 'valid');
    end
    
    energyFeatures = computeEnergyFeatures(filterResponses); 
    allEnergyFeatures(idx, :) = energyFeatures;
    idx = idx + 1;

     saveFilename = fullfile(netwFolderPath, replace(file.name, '.raw', '_energy.mat'));
     save(saveFilename, 'energyFeatures');  % 或'reducedFeatures'，如果你应用了PCA
end

% [coeff, score, latent, tsquared, explained] = pca(allEnergyFeatures);

% reducedFeatures = score(:, 1:3);
% colors = {'r','r','r','r','r','r','r','r','r', 'g', 'g', 'g', 'g', 'g', 'g', 'g', 'g', 'g', 'b','b','b','b','b','b','b','b','b', 'm','m','m','m','m','m','m','m','m'}.'; 
% markers = {'o','o','o','o','o','o','o','o','o', '+','+','+','+','+','+','+','+','+', '*','*','*','*','*','*','*','*','*', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x'}.';
% c = [1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4 4];
% scatter3(reducedFeatures(:,1), reducedFeatures(:,2), reducedFeatures(:,3), 30, c, 'filled');

% xlabel('Principal Component 1');
% ylabel('Principal Component 2');
% zlabel('Principal Component 3');
% title('3D Feature Space');

% for idx = 1:numImages
%     saveFilename = fullfile(newFolderPath, replace(rawFiles(idx).name, '.raw', '_pca.mat'));
%     pcaFeatures = reducedFeatures(idx, :);
%     save(saveFilename, 'pcaFeatures');
% end
function energyFeatures = computeEnergyFeatures(filterResponses)
    squaredResponses = filterResponses .^ 2;
    
    energyFeatures = zeros(1, size(filterResponses, 3));
    
    for i = 1:size(filterResponses, 3)
        energy = squaredResponses(:, :, i);
        energyFeatures(i) = mean(energy(:));
    end
end

