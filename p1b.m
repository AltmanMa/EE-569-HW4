energyFolderPath = 'test_energy';
pcaFolderPath = 'test_pca'; 

energyFiles = dir(fullfile(energyFolderPath, '*.mat'));

pcaFiles = dir(fullfile(pcaFolderPath, '*.mat'));

allEnergyFeatures = [];
allPCAFeatures = [];

for i = 1:length(energyFiles)
    energyData = load(fullfile(energyFolderPath, energyFiles(i).name));
    energyFeatures = energyData.energyFeatures;
    
    allEnergyFeatures = [allEnergyFeatures; energyFeatures];
end

for i = 1:length(pcaFiles)
    pcaData = load(fullfile(pcaFolderPath, pcaFiles(i).name));
    PCAFeatures = pcaData.pcaFeatures;
    
    allPCAFeatures = [allPCAFeatures; PCAFeatures];
end

K = 4;

% Perform K-means clustering on energy features
label_25 = kmeans(allEnergyFeatures', K);

% Perform K-means clustering on PCA features
label_pca = kmeans(allPCAFeatures', K);

disp('Cluster labels for energy features:');
disp(num2str(label_25'));


disp('Cluster labels for PCA features:');
disp(num2str(label_pca'));

