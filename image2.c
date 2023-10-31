#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <string.h>
#include <omp.h>  // Include OpenMP header
#include "image.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// An array of kernel matrices to be used for image convolution.
// The indexes of these match the enumeration from the header file.
Matrix algorithms[] = {
    {{0, -1, 0}, {-1, 4, -1}, {0, -1, 0}},
    {{0, -1, 0}, {-1, 5, -1}, {0, -1, 0}},
    {{1 / 9.0, 1 / 9.0, 1 / 9.0}, {1 / 9.0, 1 / 9.0, 1 / 9.0}, {1 / 9.0, 1 / 9.0, 1 / 9.0}},
    {{1.0 / 16, 1.0 / 8, 1.0 / 16}, {1.0 / 8, 1.0 / 4, 1.0 / 8}, {1.0 / 16, 1.0 / 8, 1.0 / 16}},
    {{-2, -1, 0}, {-1, 1, 1}, {0, 1, 2}},
    {{0, 0, 0}, {0, 1, 0}, {0, 0, 0}}
};

// Define a struct to pass thread-specific data
typedef struct {
    Image* srcImage;
    Image* destImage;
    Matrix algorithm;
    int startRow;
    int endRow;
} ThreadData;

// GetPixelValue - Computes the value of a specific pixel on a specific channel using the selected convolution kernel
uint8_t getPixelValue(Image* srcImage, int x, int y, int bit, Matrix algorithm) {
    // ... (rest of the function remains the same)
}

// Modify the Convolution Function to work on a specific portion of the image
void convoluteThread(ThreadData* data) {
    for (int row = data->startRow; row < data->endRow; row++) {
        // Convolute a portion of the image
        for (int pix = 0; pix < data->srcImage->width; pix++) {
            for (int bit = 0; bit < data->srcImage->bpp; bit++) {
                data->destImage->data[Index(pix, row, data->srcImage->width, bit, data->algorithm)] = getPixelValue(data->srcImage, pix, row, bit, data->algorithm);
            }
        }
    }
}

// Modify the Usage function to include OpenMP
int Usage() {
    printf("Usage: image <filename> <type>\n\twhere type is one of (edge, sharpen, blur, gauss, emboss, identity)\n");
    return -1;
}

// GetKernelType: Converts the string name of a convolution into a value from the KernelTypes enumeration
enum KernelTypes GetKernelType(char* type) {
    if (!strcmp(type, "edge")) return EDGE;
    else if (!strcmp(type, "sharpen")) return SHARPEN;
    else if (!strcmp(type, "blur")) return BLUR;
    else if (!strcmp(type, "gauss")) return GAUSE_BLUR;
    else if (!strcmp(type, "emboss")) return EMBOSS;
    else return IDENTITY;
}

int main(int argc, char** argv) {
    long t1, t2;
    t1 = time(NULL);

    stbi_set_flip_vertically_on_load(0);
    if (argc != 3) return Usage();
    char* fileName = argv[1];
    if (!strcmp(argv[1], "pic4.jpg") && !strcmp(argv[2], "gauss")) {
        printf("You have applied a Gaussian filter to Gauss which has caused a tear in the time-space continuum.\n");
    }
    enum KernelTypes type = GetKernelType(argv[2]);

    Image srcImage, destImage, bwImage;
    srcImage.data = stbi_load(fileName, &srcImage.width, &srcImage.height, &srcImage.bpp, 0);
    if (!srcImage.data) {
        printf("Error loading file %s.\n", fileName);
        return -1;
    }
    destImage.bpp = srcImage.bpp;
    destImage.height = srcImage.height;
    destImage.width = srcImage.width;
    destImage.data = malloc(sizeof(uint8_t) * destImage.width * destImage.bpp * destImage.height);

    // Parallelize the convolution using OpenMP
    int numThreads = 4;  // You can adjust the number of threads
    int rowsPerThread = srcImage.height / numThreads;

    #pragma omp parallel num_threads(numThreads)
    {
        int threadId = omp_get_thread_num();
        ThreadData threadData;
        threadData.srcImage = &srcImage;
        threadData.destImage = &destImage;
        threadData.algorithm = algorithms[type];
        threadData.startRow = threadId * rowsPerThread;
        threadData.endRow = (threadId == numThreads - 1) ? srcImage.height : (threadId + 1) * rowsPerThread;

        convoluteThread(&threadData);
    }

    stbi_write_png("output.png", destImage.width, destImage.height, destImage.bpp, destImage.data, destImage.bpp * destImage.width);
    stbi_image_free(srcImage.data);

    free(destImage.data);
    t2 = time(NULL);
    printf("Took %ld seconds\n", t2 - t1);
    return 0;
}
