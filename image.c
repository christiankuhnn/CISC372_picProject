#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <string.h>
#include "image.h"
#include <pthread.h>
#include <stdlib.h>
#include "stb_image.h"
#include "stb_image_write.h"

#define NUM_THREADS 4

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

// Function to compute the value of a specific pixel on a specific channel using the selected convolution kernel
uint8_t getPixelValue(Image* srcImage, int x, int y, int bit, Matrix algorithm) {
    int px, mx, py, my;
    int span = srcImage->width * srcImage->bpp;
    px = x + 1; py = y + 1; mx = x - 1; my = y - 1;

    // Handle edge pixels
    if (mx < 0) mx = 0;
    if (my < 0) my = 0;
    if (px >= srcImage->width) px = srcImage->width - 1;
    if (py >= srcImage->height) py = srcImage->height - 1;

    uint8_t result =
        algorithm[0][0] * srcImage->data[Index(mx, my, srcImage->width, bit, srcImage->bpp)] +
        algorithm[0][1] * srcImage->data[Index(x, my, srcImage->width, bit, srcImage->bpp)] +
        algorithm[0][2] * srcImage->data[Index(px, my, srcImage->width, bit, srcImage->bpp)] +
        algorithm[1][0] * srcImage->data[Index(mx, y, srcImage->width, bit, srcImage->bpp)] +
        algorithm[1][1] * srcImage->data[Index(x, y, srcImage->width, bit, srcImage->bpp)] +
        algorithm[1][2] * srcImage->data[Index(px, y, srcImage->width, bit, srcImage->bpp)] +
        algorithm[2][0] * srcImage->data[Index(mx, py, srcImage->width, bit, srcImage->bpp)] +
        algorithm[2][1] * srcImage->data[Index(x, py, srcImage->width, bit, srcImage->bpp)] +
        algorithm[2][2] * srcImage->data[Index(px, py, srcImage->width, bit, srcImage->bpp)];

    return result;
}

// Function to apply a kernel matrix to an image
void convolute(Image* srcImage, Image* destImage, Matrix algorithm, int startRow, int endRow) {
    for (int row = startRow; row < endRow; row++) {
        for (int pix = 0; pix < srcImage->width; pix++) {
            for (int bit = 0; bit < srcImage->bpp; bit++) {
                destImage->data[Index(pix, row, srcImage->width, bit, srcImage->bpp)] = getPixelValue(srcImage, pix, row, bit, algorithm);
            }
        }
    }
}

int Usage() {
    printf("Usage: image <filename> <type>\n\twhere type is one of (edge, sharpen, blur, gauss, emboss, identity)\n");
    return -1;
}

enum KernelTypes GetKernelType(char* type) {
    if (!strcmp(type, "edge")) return EDGE;
    else if (!strcmp(type, "sharpen")) return SHARPEN;
    else if (!strcmp(type, "blur")) return BLUR;
    else if (!strcmp(type, "gauss")) return GAUSE_BLUR;
    else if (!strcmp(type, "emboss")) return EMBOSS;
    else return IDENTITY;
}

typedef struct {
    int thread_id;
    Image* srcImage;
    Image* destImage;
    Matrix algorithm;
    int startRow;
    int endRow;
} ThreadData;

void* convoluteThread(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    convolute(data->srcImage, data->destImage, data->algorithm, data->startRow, data->endRow);
    pthread_exit(NULL);
}

int main(int argc, char** argv) {
    long t1, t2;
    t1 = time(NULL);

    stbi_set_flip_vertically_on_load(0);
    if (argc != 3) return Usage();
    char* fileName = argv[1];
    if (!strcmp(argv[1], "pic4.jpg") && !strcmp(argv[2], "gauss")) {
        printf("You have applied a gaussian filter to Gauss which has caused a tear in the time-space continuum.\n");
    }
    enum KernelTypes type = GetKernelType(argv[2]);

    Image srcImage, destImage;
    srcImage.data = stbi_load(fileName, &srcImage.width, &srcImage.height, &srcImage.bpp, 0);
    if (!srcImage.data) {
        printf("Error loading file %s.\n", fileName);
        return -1;
    }
    destImage.bpp = srcImage.bpp;
    destImage.height = srcImage.height;
    destImage.width = srcImage.width;
    destImage.data = malloc(sizeof(uint8_t) * destImage.width * destImage.bpp * destImage.height);

    // Create an array of pthreads
    pthread_t threads[NUM_THREADS];

    // Create an array of thread data
    ThreadData thread_data[NUM_THREADS];

    int rowsPerThread = srcImage.height / NUM_THREADS;
    for (int i = 0; i < NUM_THREADS; i++) {
        thread_data[i].thread_id = i;
        thread_data[i].srcImage = &srcImage;
        thread_data[i].destImage = &destImage;
        thread_data[i].startRow = i * rowsPerThread;
        thread_data[i].endRow = (i == NUM_THREADS - 1) ? srcImage.height : (i + 1) * rowsPerThread;
        memcpy(thread_data[i].algorithm, algorithms[type], sizeof(Matrix));
        pthread_create(&threads[i], NULL, convoluteThread, (void*)&thread_data[i]);
    }

    // Wait for all threads to finish
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    stbi_write_png("output.png", destImage.width, destImage.height, destImage.bpp, destImage.data, destImage.bpp * destImage.width);
    stbi_image_free(srcImage.data);
    free(destImage.data);

    t2 = time(NULL);
    printf("Took %ld seconds\n", t2 - t1);
    return 0;
}
