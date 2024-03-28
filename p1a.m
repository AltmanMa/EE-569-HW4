folderPath = 'EE569_2024Spring_HW4_materials/train';
newFolderPath = 'train_energy';

% 获取文件夹中所有.raw文件
rawFiles = dir(fullfile(folderPath, '*.raw'));

% 初始化Laws的1D滤波器
L5 = [1 4 6 4 1];
E5 = [-1 -2 0 2 1];
S5 = [-1 0 2 0 -1];
W5 = [-1 2 0 -2 1];
R5 = [1 -4 6 -4 1];

% 创建25个5x5的Laws滤波器
filterBank = {
    conv2(L5', L5), conv2(L5', E5), conv2(L5', S5), conv2(L5', W5), conv2(L5', R5),
    conv2(E5', L5), conv2(E5', E5), conv2(E5', S5), conv2(E5', W5), conv2(E5', R5),
    conv2(S5', L5), conv2(S5', E5), conv2(S5', S5), conv2(S5', W5), conv2(S5', R5),
    conv2(W5', L5), conv2(W5', E5), conv2(W5', S5), conv2(W5', W5), conv2(W5', R5),
    conv2(R5', L5), conv2(R5', E5), conv2(R5', S5), conv2(R5', W5), conv2(R5', R5)
};

% 遍历文件列表
for file = rawFiles'
    % 构建完整的文件路径
    filename = fullfile(folderPath, file.name);
    
    % 读取图像数据，这里假设所有图像都是128x128的灰度图像
    imageData = readraw(filename, 128, 128);
    
    % 应用25种Laws滤波器，获取滤波器响应（这里需要你自己实现这个函数）
    filterResponses = zeros(128, 128, length(filterBank));

    for i = 1:length(filterBank)
        filter = filterBank{i};
        % 使用边界复制进行边界扩展
        extendedImage = padarray(imageData, [2 2], 'replicate', 'both');
        % 计算响应并存储
        filterResponses(:, :, i) = conv2(extendedImage, filter, 'valid');
    end
    
    % 计算能量特征（这里需要你自己实现这个函数）
    energyFeatures = computeEnergyFeatures(filterResponses);  % 注意: 你需要实现computeEnergyFeatures函数
    
    % （可选）应用PCA（这里需要你自己实现这个函数）
    % reducedFeatures = applyPCA(energyFeatures);  % 注意: 你需要实现applyPCA函数
    
    % 根据原始文件名保存处理后的数据为.mat文件
    saveFilename = fullfile(newFolderPath, replace(file.name, '.raw', '_energy.mat'));
    save(saveFilename, 'energyFeatures');  % 或'reducedFeatures'，如果你应用了PCA
end


function energyFeatures = computeEnergyFeatures(filterResponses)
    % 计算每个滤波器响应的平方
    squaredResponses = filterResponses .^ 2;
    
    % 初始化能量特征向量
    energyFeatures = zeros(1, size(filterResponses, 3));
    
    % 对每个滤波器响应计算平均能量
    for i = 1:size(filterResponses, 3)
        % 提取第i个滤波器的响应并计算其平均能量
        energy = squaredResponses(:, :, i);
        energyFeatures(i) = mean(energy(:));
    end
end

