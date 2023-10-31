#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <string.h>
#include "image.h"
#include <pthread.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define NUM_THREADS 4

pthread_t threads[NUM_THREADS];
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

// Define the kernel matrices as double arrays
double algorithms[][3][3] = {
    {{0.0, -1.0, 0.0}, {-1.0, 4.0, -1.0}, {0.0, -1.0, 0.0}},
    {{0.0, -1.0, 0.0}, {-1.0, 5.0, -1.0}, {0.0, -1.0, 0.0}},
    {{1.0/9.0, 1.0/9.0, 1.0/9.0}, {1.0/9.0, 1.0/9.0, 1.0/9.0}, {1.0/9.0, 1.0/9.0, 1.0/9.0}},
    {{1.0/16.0, 1.0/8.0, 1.0/16.0}, {1.0/8.0, 1.0/4.0, 1.0/8.0}, {1.0/16.0, 1.0/8.0, 1.0/16.0}},
    {{-2.0, -1.0, 0.0}, {-1.0, 1.0, 1.0}, {0.0, 1.0, 2.0}},
    {{0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 0.0}}
};

typedef struct {
    int thread_id;
    int start_row;
    int end_row;
    Image* srcImage;
    Image* destImage;
    double (*algorithm)[3][3];
} ThreadData;

void* processImage(void* arg) {
    ThreadData* data = (ThreadData*)arg;

    for (int row = data->start_row; row < data->end_row; row++) {
        for (int pix = 0; pix < data->srcImage->width; pix++) {
            for (int bit = 0; bit < data->srcImage->bpp; bit++) {
                uint8_t pixelValue = getPixelValue(data->srcImage, pix, row, bit, *(data->algorithm));

                pthread_mutex_lock(&mutex);
                data->destImage->data[Index(pix, row, data->srcImage->width, bit, data->srcImage->bpp)] = pixelValue;
                pthread_mutex_unlock(&mutex);
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
    enum KernelTypes type = GetKernelType(argv[2]);

    Image srcImage, *destImage;
    srcImage.data = stbi_load(fileName, &srcImage.width, &srcImage.height, &srcImage.bpp, 0);

    if (!srcImage.data) {
        printf("Error loading file %s.\n", fileName);
        return -1;
    }

    destImage = (Image*)malloc(sizeof(Image));
    destImage->bpp = srcImage.bpp;
    destImage->height = srcImage.height;
    destImage->width = srcImage.width;
    destImage->data = (uint8_t*)malloc(sizeof(uint8_t) * destImage->width * destImage->bpp * destImage->height);

    int rows_per_thread = srcImage.height / NUM_THREADS;

    for (int i = 0; i < NUM_THREADS; i++) {
        ThreadData* thread_data = (ThreadData*)malloc(sizeof(ThreadData));
        thread_data->thread_id = i;
        thread_data->start_row = i * rows_per_thread;
        thread_data->end_row = (i == NUM_THREADS - 1) ? srcImage.height : (i + 1) * rows_per_thread;
        thread_data->srcImage = &srcImage;
        thread_data->destImage = destImage;
        thread_data->algorithm = algorithms[type];

        pthread_create(&threads[i], NULL, processImage, (void*)thread_data);
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    stbi_write_png("output.png", destImage->width, destImage->height, destImage->bpp, destImage->data, destImage->bpp * destImage->width);
    stbi_image_free(srcImage.data);

    free(destImage->data);
    free(destImage);
    t2 = time(NULL);
    printf("Took %ld seconds\n", t2 - t1);
    return 0;
}
