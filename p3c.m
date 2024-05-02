Cat1 = imread('EE569_2024Spring_HW4_materials/cat_1.png');
Cat2 = imread('EE569_2024Spring_HW4_materials/cat_2.png');
Cat3 = imread('EE569_2024Spring_HW4_materials/cat_3.png');
Dog1 = imread('EE569_2024Spring_HW4_materials/dog_1.png');

Cat1 = single(rgb2gray(Cat1));
Cat2 = single(rgb2gray(Cat2));
Cat3 = single(rgb2gray(Cat3));
Dog1 = single(rgb2gray(Dog1));

[f_cat1, d_cat1] = vl_sift(Cat1);
[f_cat2, d_cat2] = vl_sift(Cat2);
[f_cat3, d_cat3] = vl_sift(Cat3);
[f_dog1, d_dog1] = vl_sift(Dog1);

all_sift_features = [d_cat1'; d_cat2'; d_cat3'; d_dog1'];

num_clusters = 8;
[cluster_indices, centroids] = kmeans(double(all_sift_features), num_clusters);

codewords_cat1 = histcounts(cluster_indices(1:size(d_cat1, 2)), 1:num_clusters+1);
codewords_cat2 = histcounts(cluster_indices(size(d_cat1, 2)+1:size(d_cat1, 2)+size(d_cat2, 2)), 1:num_clusters+1);
codewords_cat3 = histcounts(cluster_indices(size(d_cat1, 2)+size(d_cat2, 2)+1:size(d_cat1, 2)+size(d_cat2, 2)+size(d_cat3, 2)), 1:num_clusters+1);
codewords_dog1 = histcounts(cluster_indices(size(d_cat1, 2)+size(d_cat2, 2)+size(d_cat3, 2)+1:end), 1:num_clusters+1);


similarity_to_cat1 = norm(codewords_cat3 - codewords_cat1);
similarity_to_cat2 = norm(codewords_cat3 - codewords_cat2);
similarity_to_dog1 = norm(codewords_cat3 - codewords_dog1);

disp('Similarity of Cat_3 codewords to other images:');
disp(['Similarity to Cat_1: ', num2str(similarity_to_cat1)]);
disp(['Similarity to Cat_2: ', num2str(similarity_to_cat2)]);
disp(['Similarity to Dog_1: ', num2str(similarity_to_dog1)]);

