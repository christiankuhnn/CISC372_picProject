
image:image.c image.h
	gcc image.c -o image -lstb_image -lpthread -lm

clean:
	rm -f image output.png