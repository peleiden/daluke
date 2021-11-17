CC=gcc
CFLAGS=-fPIC -Wall -O3
LDFLAGS=-shared

all: so/masking.so

daluke/c/masking.o: daluke/c/masking.c

so/masking.so: daluke/c/masking.o
	mkdir -p so
	$(CC) daluke/c/masking.o -shared -fPIC -o so/masking.so
	$(RM) daluke/c/*.o

clean:
	$(RM) daluke/c/masking/*.o so/*.so
