#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <string.h>
#include <omp.h>
#include "image.h"
#include "stb_image.h"
#include "stb_image_write.h"

Matrix algorithms[] = {
    {{0, -1, 0}, {-1, 4, -1}, {0, -1, 0}},
    {{0, -1, 0}, {-1, 5, -1}, {0, -1, 0}},
    {{1/9.0, 1/9.0, 1/9.0}, {1/9.0, 1/9.0, 1/9.0}, {1/9.0, 1/9.0, 1/9.0}},
    {{1.0/16, 1.0/8, 1.0/16}, {1.0/8, 1.0/4, 1.0/8}, {1.0/16, 1.0/8, 1.0/16}},
    {{-2, -1, 0}, {-1, 1, 1}, {0, 1, 2}},
    {{0, 0, 0}, {0, 1, 0}, {0, 0, 0}}
};

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

    #pragma omp parallel
    {
        #pragma omp for
        for (int row = 0; row < srcImage.height; row++) {
            for (int pix = 0; pix < srcImage.width; pix++) {
                for (int bit = 0; bit < srcImage.bpp; bit++) {
                    destImage.data[Index(pix, row, srcImage.width, bit, srcImage.bpp)] =
                        getPixelValue(&srcImage, pix, row, bit, algorithms[type]);
                }
            }
        }
    }

    stbi_write_png("output.png", destImage.width, destImage.height, destImage.bpp, destImage.data, destImage.bpp * destImage.width);
    stbi_image_free(srcImage.data);

    free(destImage.data);
    t2 = time(NULL);
    printf("Took %ld seconds\n", t2 - t1);
    return 0;
}
