#include <stdio.h>
#include <vector>
#include <iostream>
#include <stdlib.h>

using namespace std;

// Assuming the Laws filters and PCA are already implemented and available
std::vector<int> L5 ={1,4,6,4,1};
std::vector<int> E5 ={-1,-2,0,2,1};
std::vector<int> S5 ={-1,0,2,0,-1};
std::vector<int> W5 ={-1,2,0,-2,1};
std::vector<int> R5 ={1,-4,6,-4,1};

std::vector<std::vector<int>> outerProduct(const std::vector<int>& a, const std::vector<int>& b) {
    std::vector<std::vector<int>> product(a.size(), std::vector<int>(b.size()));
    for (size_t i = 0; i < a.size(); ++i) {
        for (size_t j = 0; j < b.size(); ++j) {
            product[i][j] = a[i] * b[j];
        }
    }
    return product;
}

std::vector<std::vector<int>> LL = outerProduct(L5, L5);
std::vector<std::vector<int>> LE = outerProduct(L5, E5);
std::vector<std::vector<int>> LS = outerProduct(L5, S5);
std::vector<std::vector<int>> LW = outerProduct(L5, W5);
std::vector<std::vector<int>> LR = outerProduct(L5, R5);
std::vector<std::vector<int>> EL = outerProduct(E5, L5);
std::vector<std::vector<int>> EE = outerProduct(E5, E5);
std::vector<std::vector<int>> ES = outerProduct(E5, S5);
std::vector<std::vector<int>> EW = outerProduct(E5, W5);
std::vector<std::vector<int>> ER = outerProduct(E5, R5);
std::vector<std::vector<int>> SL = outerProduct(S5, L5);
std::vector<std::vector<int>> SE = outerProduct(S5, E5);
std::vector<std::vector<int>> SS = outerProduct(S5, S5);
std::vector<std::vector<int>> SW = outerProduct(S5, W5);
std::vector<std::vector<int>> SR = outerProduct(S5, R5);
std::vector<std::vector<int>> WL = outerProduct(W5, L5);
std::vector<std::vector<int>> WE = outerProduct(W5, E5);
std::vector<std::vector<int>> WS = outerProduct(W5, S5);
std::vector<std::vector<int>> WW = outerProduct(W5, W5);
std::vector<std::vector<int>> WR = outerProduct(W5, R5);
std::vector<std::vector<int>> RL = outerProduct(R5, L5);
std::vector<std::vector<int>> RE = outerProduct(R5, E5);
std::vector<std::vector<int>> RS = outerProduct(R5, S5);
std::vector<std::vector<int>> RW = outerProduct(R5, W5);
std::vector<std::vector<int>> RR = outerProduct(R5, R5);


// #include "LawsFilters.h"
// #include "PCA.h"

int main(int argc, char *argv[])
{
    FILE *file;
    int BytesPerPixel = 1; // Since the images are grayscale
    int Size = 128;       // The images are 128x128

    // Check for proper syntax
    if (argc < 3){
        cout << "Syntax Error - Incorrect Parameter Usage:" << endl;
        cout << "program_name input_image.raw output_feature_vector.txt" << endl;
        return 0;
    }

    // Allocate memory for the input image
    unsigned char Imagedata[Size][Size][BytesPerPixel];

    // Read image into Imagedata matrix
    if (!(file = fopen(argv[1], "rb"))) {
        cout << "Cannot open file: " << argv[1] << endl;
        exit(1);
    }
    fread(Imagedata, sizeof(unsigned char), Size * Size * BytesPerPixel, file);
    fclose(file);

    // Here, you would apply the Laws filters to the image
    // and compute the feature vector for the image.
    // This is highly application-specific and would require
    // you to write the code or use a library to perform this step.
    // For instance:
    // vector<vector<double>> featureVector = ApplyLawsFilters(Imagedata, Size, BytesPerPixel);

    // After applying Laws filters, you would calculate the energy features
    // and then average them to get a single feature vector for the image.
    // vector<double> averagedFeatureVector = ComputeAverageEnergyFeatures(featureVector);

    // Optionally, apply PCA to reduce the dimensionality of the feature vector
    // vector<double> reducedFeatureVector = ApplyPCA(averagedFeatureVector);

    // Save the feature vector to a file
    // This is just an example of how you might save the feature vector.
    // The actual implementation will depend on how you calculate the feature vector.
    if (!(file = fopen(argv[2], "wb"))) {
        cout << "Cannot open file: " << argv[2] << endl;
        exit(1);
    }
    // for (double feature : reducedFeatureVector) {
    //     fprintf(file, "%lf\n", feature);
    // }
    fclose(file);

    return 0;
}
