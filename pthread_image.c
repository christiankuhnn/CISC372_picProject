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

#define NUM_THREADS 10

typedef struct inputStruct {
  Image* srcImage;
  Image* destImage;
  enum KernelTypes type;
  long rank;
} inputStruct;

void *threaded_convolute(void *args){
  int row, pix, bit, span;
  inputStruct* input = (inputStruct*)args;
  int localHeight = input->srcImage->height / NUM_THREADS;
  int localHeightRemainder = input->srcImage->height % NUM_THREADS;
  int start_row = input->rank * localHeight;
  int end_row = (input->rank + 1) * localHeight;
  if (input->rank == NUM_THREADS - 1)
    end_row = end_row + localHeightRemainder;

  for (row = start_row; row < end_row; row++) {
    for (pix = 0; pix < input->srcImage->width; pix++) {
      for (bit = 0; bit < input->srcImage->bpp; bit++) {
        input->destImage->data[Index(pix, row, input->srcImage->width, bit, input->srcImage->bpp)] =
            getPixelValue(input->srcImage, pix, row, bit, algorithms[input->type]);
      }
    }
  }
}



int main(int argc, char** argv) {
    long t1, t2, thread;
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

    pthread_t* thread_handlers = (pthread_t*) malloc(NUM_THREADS * sizeof(pthread_t));
    for(thread = 0; thread < NUM_THREADS; thread++){
      inputStruct *params = (inputStruct*)malloc(sizeof(inputStruct));
      params->srcImage = &srcImage;
      params->destImage = &destImage;
      params->type = type;
      params->rank = thread;
      pthread_create(&thread_handlers[thread], NULL, &threaded_convolute, (void*)params);
    }
    for (thread = 0; thread < NUM_THREADS; thread++){
        pthread_join(thread_handlers[thread], NULL);
    }
    free(thread_handlers);

    stbi_write_png("output.png", destImage.width, destImage.height, destImage.bpp, destImage.data, destImage.bpp * destImage.width);
    stbi_image_free(srcImage.data);

    free(destImage.data);
    t2 = time(NULL);
    printf("Took %ld seconds\n", t2 - t1);
   return 0;
}
