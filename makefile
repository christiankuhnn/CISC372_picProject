image: image.c image.h
	gcc -g image.c -o image -lm
omp: omp_image.c image.h
	gcc -g -fopenmp omp_image.c -o image -lm
pthread: pthread_image.c image.h
	gcc -g -lpthread pthread_image.c -o image -lm
clean:
	rm -f image output.png