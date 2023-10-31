#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <pthread.h>
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
    {{1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0}, {1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0}, {1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0}},
    {{1.0 / 16, 1.0 / 8, 1.0 / 16}, {1.0 / 8, 1.0 / 4, 1.0 / 8}, {1.0 / 16, 1.0 / 8, 1.0 / 16}},
    {{-2, -1, 0}, {-1, 1, 1}, {0, 1, 2}},
    {{0, 0, 0}, {0, 1, 0}, {0, 0, 0}}
};

// Mutex for thread synchronization
pthread_mutex_t mutex;

// Thread data structure
typedef struct {
    Image* srcImage;
    Image* destImage;
    int start_row;
    int end_row;
    enum KernelTypes type;
} ThreadData;

// getPixelValue - Computes the value of a specific pixel on a specific channel using the selected convolution kernel
uint8_t getPixelValue(Image* srcImage, int x, int y, int bit, Matrix algorithm) {
    int px, mx, py, my;
    uint8_t result = 0;
    int span = srcImage->width * srcImage->bpp;

    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            px = x + j;
            py = y + i;
            mx = x - j;
            my = y - i;

            if (mx < 0) mx = 0;
            if (my < 0) my = 0;
            if (px >= srcImage->width) px = srcImage->width - 1;
            if (py >= srcImage->height) py = srcImage->height - 1;

            result += algorithm[i + 1][j + 1] * srcImage->data[Index(mx, my, srcImage->width, bit, srcImage->bpp)];
        }
    }

    return result;
}

// Modified convolute function for multi-threading
void* applyConvolution(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    int start_row = data->start_row;
    int end_row = data->end_row;
    Image* srcImage = data->srcImage;
    Image* destImage = data->destImage;
    enum KernelTypes type = data->type;

    for (int row = start_row; row < end_row; row++) {
        for (int pix = 0; pix < srcImage->width; pix++) {
            for (int bit = 0; bit < srcImage->bpp; bit++) {
                destImage->data[Index(pix, row, srcImage->width, bit, srcImage->bpp)] = getPixelValue(srcImage, pix, row, bit, algorithms[type]);
            }
        }
    }

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

    // Initialize the mutex
    pthread_mutex_init(&mutex, NULL);

    // Split the image into equal parts for multi-threading
    int num_threads = 4;
    int rows_per_thread = srcImage.height / num_threads;
    pthread_t threads[num_threads];
    ThreadData thread_data[num_threads];

    // Create and manage threads
    for (int i = 0; i < num_threads; i++) {
        thread_data[i].srcImage = &srcImage;
        thread_data[i].destImage = &destImage;
        thread_data[i].start_row = i * rows_per_thread;
        thread_data[i].end_row = (i == num_threads - 1) ? srcImage.height : (i + 1) * rows_per_thread;
        thread_data[i].type = type;

        pthread_create(&threads[i], NULL, applyConvolution, &thread_data[i]);
    }

    // Wait for threads to finish
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    // Cleanup the mutex
    pthread_mutex_destroy(&mutex);

    stbi_write_png("output.png", destImage.width, destImage.height, destImage.bpp, destImage.data, destImage.bpp * destImage.width);
    stbi_image_free(srcImage.data);

    free(destImage.data);
    t2 = time(NULL);
    printf("Took %ld seconds\n", t2 - t1);
    return 0;
}
