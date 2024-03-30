trainEnergyFolderPath = 'train_energy';
trainPCAFolderPath = 'train_pca'; 
testEnergyFolderPath = 'test_energy'; 
testPCAFolderPath = 'test_pca'; 

trainLabels = [ones(9, 1); 2 * ones(9, 1); 3 * ones(9, 1); 4 * ones(9, 1)];

testLabels = [2 1 4 1 4 3 3 4 2 2 1 3];

% Initialize arrays to store predicted labels
predictedLabels25D = zeros(length(testLabels), 1);
predictedLabels3D = zeros(length(testLabels), 1);

% Train and predict using 25-D features
trainFeatures25D = [];
trainFiles25D = dir(fullfile(trainEnergyFolderPath, '*.mat'));
for i = 1:length(trainFiles25D)
    trainData = load(fullfile(trainEnergyFolderPath, trainFiles25D(i).name));
    trainFeatures25D = [trainFeatures25D; trainData.energyFeatures];
end

% Train SVM model
svmModel25D = fitcecoc(trainFeatures25D, trainLabels);

% Load and predict using test features
testFiles25D = dir(fullfile(testEnergyFolderPath, '*.mat'));
for i = 1:length(testFiles25D)
    testData = load(fullfile(testEnergyFolderPath, testFiles25D(i).name));
    testFeatures25D = testData.energyFeatures;
    predictedLabels25D(i) = predict(svmModel25D, testFeatures25D);
end

% Calculate error rate for 25-D features
errorRate25D = sum(predictedLabels25D ~= testLabels) / length(testLabels);
disp(['Error rate for 25-D features: ', num2str(errorRate25D)]);

% Train and predict using reduced 3-D features via PCA
trainFeatures3D = [];
trainFiles3D = dir(fullfile(trainPCAFolderPath, '*.mat'));
for i = 1:length(trainFiles3D)
    trainData = load(fullfile(trainPCAFolderPath, trainFiles3D(i).name));
    trainFeatures3D = [trainFeatures3D; trainData.pcaFeatures];
end

% Train SVM model
svmModel3D = fitcecoc(trainFeatures3D, trainLabels);

% Load and predict using test features
testFiles3D = dir(fullfile(testPCAFolderPath, '*.mat'));
for i = 1:length(testFiles3D)
    testData = load(fullfile(testPCAFolderPath, testFiles3D(i).name));
    testFeatures3D = testData.pcaFeatures;
    predictedLabels3D(i) = predict(svmModel3D, testFeatures3D);
end

% Calculate error rate for 3-D features via PCA
errorRate3D = sum(predictedLabels3D ~= testLabels) / length(testLabels);
disp(['Error rate for 3-D features via PCA: ', num2str(errorRate3D)]);

