Cat1 = imread('EE569_2024Spring_HW4_materials/cat_1.png');
Cat2 = imread('EE569_2024Spring_HW4_materials/cat_2.png');
Cat3 = imread('EE569_2024Spring_HW4_materials/cat_3.png');
Dog1 = imread('EE569_2024Spring_HW4_materials/dog_1.png');

Cat1_gray = rgb2gray(Cat1);
Cat2_gray = rgb2gray(Cat2);
Cat3_gray = rgb2gray(Cat3);
Dog1_gray = rgb2gray(Dog1);

[f1, d1] = vl_sift(single(Cat1_gray));
[f2, d2] = vl_sift(single(Cat2_gray));
[f3, d3] = vl_sift(single(Cat3_gray));
[f4, d4] = vl_sift(single(Dog1_gray));

%do the matches for the four pairs
[matches1, scores1] = vl_ubcmatch(d1, d3);
[matches2, scores2] = vl_ubcmatch(d3, d2);
[matches3, scores3] = vl_ubcmatch(d4, d3);
[matches4, scores4] = vl_ubcmatch(d1, d4);

%find max scale point in Cat1
[max_scale1, max_scale_idx1] = max(f1(3, :)); 
max_scale_keypoint1 = f1(1:2, max_scale_idx1); 

distances1 = pdist2(max_scale_keypoint1', f3(1:2, :)'); 
[min_dist1, min_dist_idx1] = min(distances1); 
closest_neighbor_keypoint3 = f3(1:2, min_dist_idx1); 

figure;
subplot(1, 2, 1);
imshow(Cat1);
hold on;
plot(max_scale_keypoint1(1), max_scale_keypoint1(2), 'r*');
title('Key Point with max scale in Cat1');

subplot(1, 2, 2);
imshow(Cat3);
hold on;
plot(closest_neighbor_keypoint3(1), closest_neighbor_keypoint3(2), 'r*');
title('closest neighboring key point in Cat_3');

figure;
subplot(1,2,1);
imshow(Cat1);
title('Cat1');
hold on;
plot(f1(1,matches1(1,:)),f1(2,matches1(1,:)),'ro');
subplot(1,2,2);
imshow(Cat3);
title('Cat3');
hold on;
plot(f3(1,matches1(2,:)),f3(2,matches1(2,:)),'ro');

%find max scale point in Cat3
[max_scale3, max_scale_idx3] = max(f3(3, :)); 
max_scale_keypoint3 = f3(1:2, max_scale_idx3); 

distances3 = pdist2(max_scale_keypoint3', f2(1:2, :)'); 
[min_dist3, min_dist_idx3] = min(distances3); 
closest_neighbor_keypoint2 = f2(1:2, min_dist_idx3); 

figure;
subplot(1, 2, 1);
imshow(Cat3);
hold on;
plot(max_scale_keypoint3(1), max_scale_keypoint3(2), 'r*');
title('Key Point with max scale in Cat3');

subplot(1, 2, 2);
imshow(Cat2);
hold on;
plot(closest_neighbor_keypoint2(1), closest_neighbor_keypoint2(2), 'r*');
title('closest neighboring key point in Cat_2');

figure;
subplot(1,2,1);
imshow(Cat3);
title('Cat3');
hold on;
plot(f3(1,matches2(1,:)),f3(2,matches2(1,:)),'ro');
subplot(1,2,2);
imshow(Cat2);
title('Cat2');
hold on;
plot(f2(1,matches2(2,:)),f2(2,matches2(2,:)),'ro');

%find max scale point in Dog1
[max_scale4, max_scale_idx4] = max(f4(3, :)); 
max_scale_keypoint4 = f4(1:2, max_scale_idx4); 

distances4 = pdist2(max_scale_keypoint3', f3(1:2, :)'); 
[min_dist4, min_dist_idx4] = min(distances4); 
closest_neighbor_keypoint34 = f3(1:2, min_dist_idx4); 

figure;
subplot(1, 2, 1);
imshow(Dog1);
hold on;
plot(max_scale_keypoint4(1), max_scale_keypoint4(2), 'r*');
title('Key Point with max scale in Dog1');

subplot(1, 2, 2);
imshow(Cat3);
hold on;
plot(closest_neighbor_keypoint34(1), closest_neighbor_keypoint34(2), 'r*');
title('closest neighboring key point in Cat_3');

figure;
subplot(1,2,1);
imshow(Dog1);
title('Dog1');
hold on;
plot(f4(1,matches3(1,:)),f4(2,matches3(1,:)),'ro');
subplot(1,2,2);
imshow(Cat3);
title('Cat3');
hold on;
plot(f3(1,matches3(2,:)),f3(2,matches3(2,:)),'ro');
